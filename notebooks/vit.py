from typing import Union
from typing import Optional

import torch
import torch.nn as nn

from base import BaseModel
from loss import combined_quality_loss



class PatchEmbed(BaseModel):
    def __init__(
        self, 
        img_size: tuple[int, int] = (240, 256),
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 768,
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size[0] // patch_size) * (img_size[1] // patch_size)
        
        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)  # (B, E, H', W')
        x = x.flatten(2)  # (B, E, N)
        x = x.transpose(1, 2)  # (B, N, E)
        return x


class MultiLayerPerceptron(BaseModel):
    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: int,
        dropout: float = 0.0
    ):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.dropout2(x)
        return x


class TransformerEncoder(BaseModel):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attention_dropout: float = 0.0
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            dim,
            num_heads=num_heads,
            dropout=attention_dropout,
            batch_first=True
        )
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MultiLayerPerceptron(
            dim,
            int(dim * mlp_ratio),
            dim,
            dropout
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x), 
                         attn_mask=mask)[0]
        x = x + self.mlp(self.norm2(x))
        return x


class ViTEncoder(BaseModel):
    def __init__(
        self,
        img_size: tuple[int, int] = (240, 256),
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        latent_dim: int = 1024
    ):
        super().__init__()
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim
        )
        
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.patch_embed.n_patches + 1, embed_dim)
        )
        self.pos_drop = nn.Dropout(dropout)

        self.blocks = nn.Sequential(*[
            TransformerEncoder(
                embed_dim,
                num_heads,
                mlp_ratio,
                dropout,
                attention_dropout
            )
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)
        
        # VAE components
        self.fc_mean = nn.Linear(embed_dim, latent_dim)
        self.fc_logvar = nn.Linear(embed_dim, latent_dim)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Patch embedding
        x = self.patch_embed(x)
        
        # Add cls token and position embedding
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        
        # Apply transformer blocks
        x = self.blocks(x)
        x = self.norm(x)
        
        # Use CLS token for VAE
        x = x[:, 0]
        
        # Get VAE parameters
        mu = self.fc_mean(x)
        logvar = self.fc_logvar(x)
        z = self.reparameterize(mu, logvar)
        
        return z, mu, logvar


class ViTDecoder(BaseModel):
    def __init__(
        self,
        latent_dim: int = 1024,
        img_size: tuple[int, int] = (240, 256),
        patch_size: int = 16,
        embed_dim: int = 768,
        depth: int = 8,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attention_dropout: float = 0.0
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size[0] // patch_size) * (img_size[1] // patch_size)
        
        # Project latent to sequence of patch embeddings
        self.latent_proj = nn.Linear(latent_dim, embed_dim)
        
        # Learnable position embeddings for decoder
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.n_patches, embed_dim)
        )
        
        # Transformer decoder blocks
        self.blocks = nn.Sequential(*[
            TransformerEncoder(
                embed_dim,
                num_heads,
                mlp_ratio,
                dropout,
                attention_dropout
            )
            for _ in range(depth)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        
        # Project patches to pixels
        self.pixel_proj = nn.Sequential(
            nn.Linear(embed_dim, patch_size * patch_size * 3),
            nn.GELU()
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        batch_size = z.shape[0]
        
        # Project latent to patch tokens
        x = self.latent_proj(z)
        x = x.unsqueeze(1).expand(-1, self.n_patches, -1)
        
        # Add position embeddings
        x = x + self.pos_embed
        
        # Apply transformer blocks
        x = self.blocks(x)
        x = self.norm(x)
        
        # Project to pixels
        x = self.pixel_proj(x)
        
        # Reshape to image
        x = x.reshape(
            batch_size,
            self.img_size[0] // self.patch_size,
            self.img_size[1] // self.patch_size,
            self.patch_size,
            self.patch_size,
            3
        )
        x = x.permute(0, 5, 1, 3, 2, 4).contiguous()
        x = x.reshape(batch_size, 3, self.img_size[0], self.img_size[1])
        
        return torch.sigmoid(x)


class ViTVAEModel(BaseModel):
    def __init__(
        self,
        img_size: tuple[int, int] = (240, 256),
        patch_size: int = 16,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        latent_dim: int = 1024
    ):
        super().__init__()
        self.encoder = ViTEncoder(
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            attention_dropout=attention_dropout,
            latent_dim=latent_dim
        )
        
        self.decoder = ViTDecoder(
            latent_dim=latent_dim,
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            attention_dropout=attention_dropout
        )

    def loss(
        self,
        h: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        x: torch.Tensor,
        xp: torch.Tensor,
        kl_weight: float = 0.001
    ) -> torch.Tensor:
        """Calculate VAE loss with reconstruction and KL divergence terms."""
        # Reconstruction loss using combined quality metrics
        reconstruction_loss = combined_quality_loss(
            xp, x,
            alpha=0.0,
            max_val=1.0,
            normalize_psnr=True
        )

        # KL divergence loss
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        kl_loss = kl_loss / x.size(0)

        # Total loss
        total_loss = reconstruction_loss + kl_weight * kl_loss
        
        return total_loss

    def forward(
        self, 
        x: torch.Tensor, 
        return_loss: bool = False
    ) -> Union[
        tuple[torch.Tensor, torch.Tensor],
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ]:
        # Encode
        h, mu, logvar = self.encoder(x)
        # Decode
        xp = self.decoder(h)
        
        if not return_loss:
            return xp, h
        
        return xp, h, self.loss(h, mu, logvar, x, xp)
        