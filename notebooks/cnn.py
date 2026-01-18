from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from base import BaseModel
from loss import psnr_loss


class ResidualBlock(BaseModel):
    def __init__(
        self, 
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        dropout: float = 0.1,
        use_1x1_conv: bool = True
    ) -> None:
        """Residual block with optional projection shortcut.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            stride: Stride for convolutions
            dropout: Dropout rate
            use_1x1_conv: Whether to use 1x1 conv for shortcut when channels change
        """
        super().__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels, 
                out_channels, 
                kernel_size=3, 
                stride=stride, 
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.GELU()
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                out_channels, 
                out_channels, 
                kernel_size=3, 
                stride=1, 
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout)
        )
        
        # 1x1 conv shortcut for channel/spatial dimension matching
        if use_1x1_conv:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels, 
                    out_channels, 
                    kernel_size=1, 
                    stride=stride,
                    bias=False
                ),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()
            
        self.final_activation = nn.GELU()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass adding shortcut to conv output.
        
        Args:
            x: Input tensor of shape (B, C, H, W)
            
        Returns:
            Output tensor of shape (B, C_out, H_out, W_out)
        """
        identity = self.shortcut(x)
        out = self.conv1(x)
        out = self.conv2(out)
        out = out + identity
        return self.final_activation(out)
    

class TeLU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x*F.tanh(F.exp(x))


class CNNEncoder(BaseModel):
    def __init__(self, embed_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.cnn_blocks = nn.Sequential(
            ResidualBlock(3, 8, stride=2, dropout=dropout, use_1x1_conv=True),
            ResidualBlock(8, 8, stride=1, dropout=dropout, use_1x1_conv=False),
            ResidualBlock(8, 16, stride=2, dropout=dropout, use_1x1_conv=True),
            ResidualBlock(16, 16, stride=1, dropout=dropout, use_1x1_conv=False),
            ResidualBlock(16, 32, stride=2, dropout=dropout, use_1x1_conv=True),
            ResidualBlock(32, 32, stride=1, dropout=dropout, use_1x1_conv=False),
            ResidualBlock(32, 64, stride=1, dropout=dropout, use_1x1_conv=True),
            ResidualBlock(64, 64, stride=1, dropout=dropout, use_1x1_conv=False),
        )
        self.adapter = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.LazyLinear(embed_dim, bias=False),
            nn.BatchNorm1d(embed_dim),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:        
        x = self.cnn_blocks(x)
        x = self.adapter(x)
        return x


class CNNDecoder(BaseModel):
    def __init__(self, embed_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.adapter = nn.Sequential(
            nn.LazyLinear(32, bias=False),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.Linear(32, 8*30*32, bias=False),
            nn.BatchNorm1d(8*30*32),
            nn.GELU(),
            nn.Dropout(dropout),    
        )

        self.cnn_blocks = nn.Sequential(
            nn.Sequential(
                nn.ConvTranspose2d(8, 8, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
                nn.BatchNorm2d(8),
                nn.GELU(),
            ),
            ResidualBlock(8, 16, dropout=dropout, use_1x1_conv=True),            
            ResidualBlock(16, 32, dropout=dropout, use_1x1_conv=True),
            ResidualBlock(32, 32, dropout=dropout, use_1x1_conv=False),

            # Second upsample block
            nn.Sequential(
                nn.ConvTranspose2d(32, 32, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
                nn.BatchNorm2d(32),
                nn.GELU(),
            ),
            ResidualBlock(32, 32, dropout=dropout, use_1x1_conv=False),
            ResidualBlock(32, 64, dropout=dropout, use_1x1_conv=True),
            ResidualBlock(64, 64, dropout=dropout, use_1x1_conv=False),

            # Final output block
            nn.Sequential(
                nn.ConvTranspose2d(64, 3, kernel_size=3, stride=2, padding=1, output_padding=1, bias=True),
                nn.GELU()
            ),
            ResidualBlock(3, 3, dropout=dropout, use_1x1_conv=False),
            ResidualBlock(3, 3, dropout=dropout, use_1x1_conv=False),

        )

        
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        B, D = z.shape
        x = self.adapter(z)
        x = x.view(B, 8, 30, 32)
        
        x = self.cnn_blocks(x)
        x = torch.clip(x, min=0.0, max=1.0)
        return x



class CNNEncoderVAE(BaseModel):
    def __init__(self, embed_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.encoder = CNNEncoder(dropout)
        self.fc_mu = nn.LazyLinear(embed_dim)
        self.fc_logvar = nn.LazyLinear(embed_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass through VAE encoder.
        
        Returns:
            tuple containing:
            - sampled latent vector (z)
            - mean (mu)
            - log variance (logvar)
        """
        # Encode through CNN
        x = self.encoder(x) 
        B, C, H, W = x.shape       
        x = x.reshape(B, C*H*W)
        
        # Get mu and logvar
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        
        # Sample using reparameterization trick
        z = self.reparameterize(mu, logvar)
        
        return z, mu, logvar
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick for VAE with numerical stability."""
        if self.training:
            # Clamp logvar to prevent numerical instability
            logvar = torch.clamp(logvar, min=-20, max=2)
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu
    

class CNNVAE(BaseModel):
    def __init__(self, embed_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.encoder = CNNEncoderVAE(embed_dim, dropout)
        self.decoder = CNNDecoderWithAdapter(dropout)

    def forward(
            self, x: torch.Tensor, 
            return_loss: bool = False, 
            return_mu_logvar: bool = False
        ) -> Union[tuple[torch.Tensor, torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        x = self._prepare_x(x)
        z, mu, logvar = self.encoder(x)
        x_reconstructed = self.decoder(z)

        return_values = (x_reconstructed.permute(0, 2, 3, 1)*255.0,)
        
        if return_mu_logvar:
            return_values = (*return_values, mu, logvar)

        if return_loss:
            return_values = (*return_values, self.loss(x, x_reconstructed, mu, logvar))
        
        return return_values
    
    def _prepare_x(self, x: torch.Tensor) -> torch.Tensor:
        return x.permute(0, 3, 1, 2) / 255.0
    
    def _prepare_reconstructed(self, x: torch.Tensor) -> torch.Tensor:
        return x.permute(0, 2, 3, 1).contiguous()*255.0
    
    def sample(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder.reparameterize(mu, logvar))    
    
    def generate(self, n_samples: int) -> torch.Tensor:
        z = torch.randn(n_samples, self.encoder.embed_dim)
        return self.decode(z)  
    
    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:        
        x = x.permute(0, 3, 1, 2) / 255.0
        return self.encoder(x)
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z).permute(0, 2, 3, 1).contiguous()*255.0

    def kl_divergence_loss(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Compute KL divergence between N(mu, var) and N(0, 1)
        KL(N(mu, var) || N(0, 1)) = 
            0.5 * mean(1 + log(var) - mu^2 - var)
        
        Args:
            mu: Mean tensor of shape (batch_size, latent_dim)
            logvar: Log variance tensor of shape (batch_size, latent_dim)
            
        Returns:
            Average KL divergence across the batch
        """
        # Clamp logvar for numerical stability
        logvar = torch.clamp(logvar, min=-20, max=2)
        
        kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
        return kl_div.mean()

    def loss(
        self, 
        x: torch.Tensor, 
        x_reconstructed: torch.Tensor, 
        mu: torch.Tensor, 
        logvar: torch.Tensor,
        psnr_max: float = 50.0
    ) -> torch.Tensor:
        recon_loss = psnr_loss(x_reconstructed, x)
        recon_loss = 1.0 - torch.clamp(recon_loss, min=0.0, max=psnr_max) / psnr_max

        kl_loss = self.kl_divergence_loss(mu, logvar)        
        print(f"recon_loss: {recon_loss.item()}, kl_loss: {kl_loss.item()}")
        return recon_loss + 0.01*kl_loss



class AutoEncoder(BaseModel):
    def __init__(self, embed_dim: int = 128, dropout: float = 0.1, decorrelation: bool = False):
        super().__init__()
        self.decorrelation = decorrelation
        self.encoder = CNNEncoder(embed_dim, dropout)
        self.decoder = CNNDecoder(embed_dim, dropout)

    def forward(
        self, x: torch.Tensor, 
        return_loss: bool = False,             
    ) -> Union[tuple[torch.Tensor], tuple[torch.Tensor, torch.Tensor]]:
        x = self._prepare_x(x)
        z = self.encoder(x)
        x_reconstructed = self.decoder(z)

        return_values = (self._prepare_reconstruction(x_reconstructed),)
        
        if return_loss:
            return_values = (*return_values, self.loss(x, x_reconstructed, z))
        
        return return_values
    
    def _prepare_x(self, x: torch.Tensor) -> torch.Tensor:
        return x.permute(0, 3, 1, 2) / 255.0
    
    def _prepare_reconstruction(self, x: torch.Tensor) -> torch.Tensor:
        return x.permute(0, 2, 3, 1).contiguous()*255.0
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 3, 1, 2) / 255.0
        return self.encoder(x)
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z).permute(0, 2, 3, 1).contiguous()*255.0

    def loss(
        self, 
        x: torch.Tensor, 
        x_reconstructed: torch.Tensor, 
        z: torch.Tensor,
        psnr_max: float = 50.0
    ) -> torch.Tensor:
        recon_loss = psnr_loss(x_reconstructed, x)
        recon_loss = 1.0 - torch.clamp(recon_loss, min=0.0, max=psnr_max) / psnr_max

        decorrelation_loss = 0.0
        if self.decorrelation:
            corr = torch.corrcoef(z)
            corr = corr*(1-torch.eye(corr.shape[0], device=corr.device))
            decorrelation_loss = corr.mean()

        # Only use item if the losses have gradients        
        a = recon_loss.item() if isinstance(recon_loss, torch.Tensor) else recon_loss
        b = decorrelation_loss.item() if isinstance(decorrelation_loss, torch.Tensor) else decorrelation_loss
        print(f"recon_loss: {a}, decorrelation_loss: {b}")

        return recon_loss + decorrelation_loss


class ResidualBlock(nn.Module):
    def __init__(
        self, 
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        dropout: float = 0.1,
        use_1x1_conv: bool = True
    ) -> None:
        """Residual block with optional projection shortcut.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            stride: Stride for convolutions
            dropout: Dropout rate
            use_1x1_conv: Whether to use 1x1 conv for shortcut when channels change
        """
        super().__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels, 
                out_channels, 
                kernel_size=3, 
                stride=stride, 
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.GELU()
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                out_channels, 
                out_channels, 
                kernel_size=3, 
                stride=1, 
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout)
        )
        
        # 1x1 conv shortcut for channel/spatial dimension matching
        if use_1x1_conv:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels, 
                    out_channels, 
                    kernel_size=1, 
                    stride=stride,
                    bias=False
                ),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()
            
        self.final_activation = nn.GELU()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass adding shortcut to conv output.
        
        Args:
            x: Input tensor of shape (B, C, H, W)
            
        Returns:
            Output tensor of shape (B, C_out, H_out, W_out)
        """
        identity = self.shortcut(x)
        out = self.conv1(x)
        out = self.conv2(out)
        out = out + identity
        return self.final_activation(out)
