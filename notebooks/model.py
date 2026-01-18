import math
from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from cnn import IJEPADecoder
from base import BaseModel
from patch import PatchEmbeddingModel
from loss import combined_quality_loss


class NextStatePredictor(BaseModel):    
    def __init__(self, embedding_dim: int, num_actions: int, num_heads: int) -> None:
        super().__init__()        
        self.num_actions = num_actions
        self.embedding_dim = embedding_dim

        self.state_encoder = PatchEmbeddingModel(
            img_size=(240, 256),
            img_channels=3,
            patch_size=16,
            embed_dim=embedding_dim,
            patch_embed_dim=32,
            num_heads=8,
            dropout=0.1,
            reduction='mean'
        )
        self.state_decoder = IJEPADecoder(embedding_dim=embedding_dim)

        self.state_action_adapter = self._build_adapter(
            out_dim=embedding_dim,
            hidden_dim=embedding_dim * 2
        )
        
        self.mha = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=num_heads,
            dropout=0.1,
            batch_first=True
        )

        self.next_state_predictor = self._build_adapter(
            out_dim=embedding_dim,
            hidden_dim=embedding_dim * 4,
            final_activation=nn.Sigmoid()
        )

        self.reward_predictor = self._build_adapter(
            out_dim=1,
            hidden_dim=embedding_dim * 4,
            final_activation=None
        )

        self.layer_norm1 = nn.LayerNorm(embedding_dim)
        self.layer_norm2 = nn.LayerNorm(embedding_dim)

    def _build_adapter(
        self, 
        out_dim: int, 
        hidden_dim: int, 
        final_activation: Optional[nn.Module] = None
    ) -> nn.Sequential:
        """Build a standard adapter network."""
        layers = [
            nn.LazyLinear(hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, out_dim),
            nn.BatchNorm1d(out_dim),
        ]
        if final_activation is not None:
            layers.append(final_activation)
        return nn.Sequential(*layers)

    def _get_pos_embeddings(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Generate sinusoidal positional embeddings."""
        positions = torch.arange(seq_len, device=device).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.embedding_dim, 2, device=device) * 
            (-math.log(10000.0) / self.embedding_dim)
        )
        pos_enc = torch.zeros(1, seq_len, self.embedding_dim, device=device)
        pos_enc[0, :, 0::2] = torch.sin(positions * div_term)
        pos_enc[0, :, 1::2] = torch.cos(positions * div_term)
        return pos_enc

    def forward(
        self, 
        states: torch.Tensor, 
        actions: torch.Tensor, 
        rewards: torch.Tensor,
        return_loss: bool = False,
        return_rewards: bool = False,
        return_reconstructed: bool = False
    ) -> Union[torch.Tensor, tuple[torch.Tensor, ...]]:
        batch_size, seq_len, height, width, channels = states.shape
        
        # Encode states
        flat_states = states.view(-1, height, width, channels).permute(0, 3, 1, 2)
        embeddings = self.state_encoder(flat_states).view(batch_size, seq_len, -1)
        
        # Prepare state-action pairs
        curr_embeddings = embeddings[:, :-1]
        next_embeddings = embeddings[:, 1:]
        action_onehot = F.one_hot(actions, num_classes=self.num_actions)
        state_action_pairs = torch.cat([curr_embeddings, action_onehot], dim=-1)

        # Process through transformer
        hidden = self.state_action_adapter(state_action_pairs.view(-1, state_action_pairs.size(-1)))
        hidden = hidden.view(batch_size, -1, self.embedding_dim)
        
        # Add positional embeddings and apply attention
        hidden = hidden + self._get_pos_embeddings(hidden.size(1), hidden.device)
        hidden = self.layer_norm1(hidden)

        causal_mask = torch.triu(
            torch.ones((seq_len, seq_len), device=hidden.device, dtype=torch.bool),
            diagonal=1
        )
        attn_out, _ = self.mha(hidden, hidden, hidden, attn_mask=causal_mask, is_causal=True)
        hidden = self.layer_norm2(hidden + attn_out)

        # Predict next states and rewards
        pred_embeddings = self.next_state_predictor(hidden.view(-1, self.embedding_dim))
        pred_embeddings = pred_embeddings.view(batch_size, -1, self.embedding_dim)
        
        reconstructed = self.state_decoder(pred_embeddings.view(-1, self.embedding_dim))
        reconstructed = reconstructed.view(batch_size, -1, channels, height, width)

        pred_rewards = self.reward_predictor(pred_embeddings.view(-1, self.embedding_dim))
        pred_rewards = pred_rewards.view(batch_size, -1)

        # Prepare return values
        outputs = (pred_embeddings[:, -1],)
        if return_reconstructed:
            outputs += (reconstructed.permute(0, 1, 3, 4, 2).contiguous(),)
        if return_rewards:
            outputs += (pred_rewards[:, -1],)
        if not return_loss:
            return outputs

        loss = self.loss(
            next_embeddings,
            pred_embeddings,
            embeddings,
            flat_states.view(batch_size, seq_len, channels, height, width)[:, 1:],
            reconstructed,
            rewards,
            pred_rewards
        )
        return (*outputs, loss)
    
    def loss(
        self,         
        target_embeddings: torch.Tensor,
        pred_embeddings: torch.Tensor, 
        state_embeddings: torch.Tensor,
        target_states: torch.Tensor,
        reconstructed_states: torch.Tensor,
        target_rewards: torch.Tensor,
        pred_rewards: torch.Tensor,
        pred_weight: float = 1.0,
        decorr_weight: float = 1.0,
        recon_weight: float = 0.5,
        reward_weight: float = 0.1
    ) -> torch.Tensor:
        """Calculate combined loss for state prediction, embedding decorrelation, reconstruction and rewards.
        
        Args:
            target_embeddings: Ground truth next state embeddings
            pred_embeddings: Predicted next state embeddings
            state_embeddings: Current state embeddings
            target_states: Ground truth next states
            reconstructed_states: Reconstructed next states
            target_rewards: Ground truth rewards
            pred_rewards: Predicted rewards
            pred_weight: Weight for prediction loss
            decorr_weight: Weight for decorrelation loss
            recon_weight: Weight for reconstruction loss
            reward_weight: Weight for reward prediction loss
            
        Returns:
            Weighted sum of all loss components
        """
        # State prediction loss
        prediction_loss = F.mse_loss(pred_embeddings, target_embeddings)
        
        # Embedding decorrelation loss
        batch_size, seq_len, emb_dim = state_embeddings.shape
        flat_embeddings = state_embeddings.view(-1, emb_dim)
        correlation_matrix = torch.corrcoef(flat_embeddings)
        decorrelation_loss = F.mse_loss(
            correlation_matrix, 
            torch.eye(correlation_matrix.shape[0], device=correlation_matrix.device)
        )
        
        # State reconstruction loss
        batch_size, seq_len, channels, height, width = reconstructed_states.shape
        reconstruction_loss = combined_quality_loss(
            reconstructed_states.reshape(-1, channels, height, width),
            target_states.reshape(-1, channels, height, width),
            ssim_window_size=16
        )

        # Reward prediction loss (using RMSE)
        reward_loss = torch.sqrt(F.mse_loss(pred_rewards, target_rewards))
        
        # Combine all losses with their respective weights
        total_loss = (
            pred_weight * prediction_loss +
            decorr_weight * decorrelation_loss +
            recon_weight * reconstruction_loss +
            reward_weight * reward_loss
        )
        
        return total_loss    