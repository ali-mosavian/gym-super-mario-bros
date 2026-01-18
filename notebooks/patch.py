from typing import Tuple
from typing import Literal

import torch
import torch.nn as nn

from base import BaseModel
from cnn import psnr_loss

class PatchEmbedder(BaseModel):
    def __init__(
        self,
        img_size: Tuple[int, int] = (240, 256),
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 768,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size, img_size[1] // patch_size)
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        
        # Patch embedding projection
        self.proj = nn.Conv2d(
            in_channels=in_channels, 
            out_channels=embed_dim,
            kernel_size=patch_size, 
            stride=patch_size
        )
        
        # 2D Positional embeddings
        self.pos_embed = nn.Parameter(
            torch.zeros(1, embed_dim, self.grid_size[0], self.grid_size[1]),
            requires_grad=True
        )
        
        # Add batch norm, activation, and dropout after projection
        self.bn = nn.BatchNorm2d(embed_dim)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout2d(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Project patches: (B, C, H, W) -> (B, E, H', W')
        x = self.proj(x)
        x = self.bn(x)
        x = self.activation(x)
        x = self.dropout(x)
        
        # Add positional embeddings
        x = x + self.pos_embed
        
        return x


class SpatialAggregator(BaseModel):
    def __init__(
        self,
        patch_embed_dim: int = 32,
        embed_dim: int = 768,
        num_heads: int = 8,
        dropout: float = 0.1,
        reduction: Literal['mean', 'concat'] = 'mean'
    ):
        super().__init__()
        self.reduction = reduction

        self.norm = nn.LayerNorm(patch_embed_dim)
        if reduction != 'concat':
            self.attention = nn.MultiheadAttention(
                patch_embed_dim, 
                num_heads,
                dropout=dropout,
                batch_first=True
            )
        
        # Final MLP for combining all patch embeddings
        self.mlp = nn.Sequential(
            nn.LazyLinear(embed_dim * 4),            
            nn.BatchNorm1d(embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim*4, embed_dim),            
            nn.BatchNorm1d(embed_dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input shape: (B, E, H, W)
        B, E, H, W = x.shape
        
        # Reshape to sequence: (B, H*W, E)
        x = x.permute(0, 2, 3, 1).reshape(B, H*W, E)

        # Apply layer norm
        x = self.norm(x)        

        
        # Reduce to single embedding
        match self.reduction:
            case 'mean':
                x, _ = self.attention(x, x, x)
                x = x.mean(dim=1)  # (B, E)
            case 'concat':
                x = x.flatten(1)
            case _:
                raise ValueError(f"Invalid reduction method: {self.reduction}")
        

        x = self.mlp(x)
        return x
    

class PatchEmbeddingModel(BaseModel):
    def __init__(
        self,
        img_size: Tuple[int, int] = (240, 256),
        img_channels: int = 3,
        patch_size: int = 16,
        embed_dim: int = 768,
        patch_embed_dim: int = 32,
        num_heads: int = 8,
        dropout: float = 0.1,
        reduction: Literal['mean', 'concat'] = 'mean'
    ):
        super().__init__()
        self.patch_embedder = PatchEmbedder(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=img_channels,
            embed_dim=patch_embed_dim,
            dropout=dropout
        )
        
        self.spatial_aggregator = SpatialAggregator(
            embed_dim=embed_dim,
            patch_embed_dim=patch_embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            reduction=reduction
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Get patch embeddings with positional information
        x = self.patch_embedder(x)
        
        # Aggregate spatial information into single embedding
        x = self.spatial_aggregator(x)
        
        return x


class PatchDecoder(BaseModel):
    def __init__(
        self,
        img_size: Tuple[int, int] = (240, 256),
        patch_size: int = 16,
        out_channels: int = 3,
        embed_dim: int = 768,
        patch_embed_dim: int = 32,
        dropout: float = 0.1,
        reduction: Literal['mean', 'concat'] = 'mean'
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size, img_size[1] // patch_size)
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.reduction = reduction

        # Project embedding back to spatial representation
        if reduction == 'mean':
            self.spatial_proj = nn.Sequential(
                nn.Linear(embed_dim, patch_embed_dim * self.num_patches),
                nn.GELU(),
                nn.Dropout(dropout)
            )
        else:  # concat
            self.spatial_proj = nn.Sequential(
                nn.Linear(embed_dim, patch_embed_dim * self.num_patches),
                nn.GELU(),
                nn.Dropout(dropout)
            )

        # Learnable position embedding
        self.pos_embed = nn.Parameter(
            torch.zeros(1, patch_embed_dim, self.grid_size[0], self.grid_size[1]),
            requires_grad=True
        )

        # Patch to image projection
        self.proj = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=patch_embed_dim,
                out_channels=out_channels,
                kernel_size=patch_size,
                stride=patch_size
            ),
            nn.BatchNorm2d(out_channels),
            nn.Sigmoid()  # Normalize output to [-1, 1] range
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input shape: (B, E)
        B = x.shape[0]
        
        # Project to patch embeddings: (B, P*E)
        x = self.spatial_proj(x)
        
        # Reshape to spatial form: (B, E, H, W)
        x = x.view(B, -1, self.grid_size[0], self.grid_size[1])
        
        # Add positional embeddings
        x = x + self.pos_embed
        
        # Project patches to image: (B, C, H, W)
        x = self.proj(x)
        
        return x
        


class PatchAutoEncoder(BaseModel):
    def __init__(
        self,
        img_size: Tuple[int, int] = (240, 256),
        img_channels: int = 3,
        patch_size: int = 16,
        embed_dim: int = 128,
        patch_embed_dim: int = 32,
        num_heads: int = 8,
        dropout: float = 0.1,
        reduction: Literal['mean', 'concat'] = 'mean'
    ):
        super().__init__()
        self.encoder = PatchEmbeddingModel(
            img_size=img_size,
            img_channels=img_channels,
            patch_size=patch_size,
            embed_dim=embed_dim,
            patch_embed_dim=patch_embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            reduction=reduction
        )
        
        self.decoder = PatchDecoder(
            img_size=img_size,
            patch_size=patch_size,
            out_channels=img_channels,
            embed_dim=embed_dim,
            patch_embed_dim=patch_embed_dim,
            dropout=dropout,
            reduction=reduction
        )

    def forward(self, x: torch.Tensor, return_loss: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        # Normalize input to [0,1] range
        x_normalized = x.permute(0, 3, 1, 2) / 255.0
        
        # Encode
        z = self.encoder(x_normalized)
        
        # Decode and scale back to [0,255] range
        xp = self.decoder(z)
        xp_restored = xp.permute(0, 2, 3, 1).contiguous()
        xp_restored = xp_restored * 255.0

        print(x_normalized.shape, xp.shape)
        if not return_loss:
            return z, xp_restored
        else:
            return z, xp_restored, self.loss(x, xp)
    
    def loss(self, x: torch.Tensor, x_reconstructed: torch.Tensor) -> torch.Tensor:
        psnr_value = psnr_loss(x_reconstructed, x)
        psnr_value = torch.clamp(psnr_value, min=0.0, max=50.0)
        return 1-psnr_value/50.0
