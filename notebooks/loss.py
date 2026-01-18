from typing import Union
from typing import Literal

import torch
import torch.nn.functional as F


# Loss function for training
def rmse_loss(
    pred: torch.Tensor, 
    target: torch.Tensor
) -> torch.Tensor:
    """Calculate RMSE loss between predicted and target observations."""
    return torch.sqrt(torch.mean((pred - target) ** 2))


def ssim_loss(
    pred: torch.Tensor, 
    target: torch.Tensor,
    window_size: int = 11,
    sigma: float = 1.5
) -> torch.Tensor:
    """Computes SSIM-based loss between images.
    
    Args:
        pred: Predicted images tensor of shape (B, C, H, W)
        target: Target images tensor of shape (B, C, H, W)
        window_size: Size of the gaussian window
        sigma: Standard deviation of gaussian window
        
    Returns:
        1 - SSIM (as a loss value)
    """
    # Create gaussian window
    gaussian = torch.exp(
        torch.tensor(
            [
                -(x - window_size//2)**2 / float(2*sigma**2) 
                for x in range(window_size)
            ],
            dtype=pred.dtype
        )
    )
    gaussian = gaussian / gaussian.sum()
    window = gaussian.unsqueeze(1) @ gaussian.unsqueeze(0)
    window = window.unsqueeze(0).unsqueeze(0)
    window = window.expand(pred.size(1), 1, window_size, window_size)
    window = window.to(pred.device)

    # Calculate SSIM
    c1 = (0.01 * 255)**2
    c2 = (0.03 * 255)**2

    mu1 = F.conv2d(pred, window, padding=window_size//2, groups=pred.shape[1])
    mu2 = F.conv2d(target, window, padding=window_size//2, groups=target.shape[1])

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(pred * pred, window, padding=window_size//2, groups=pred.shape[1]) - mu1_sq
    sigma2_sq = F.conv2d(target * target, window, padding=window_size//2, groups=target.shape[1]) - mu2_sq
    sigma12 = F.conv2d(pred * target, window, padding=window_size//2, groups=pred.shape[1]) - mu1_mu2

    ssim = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / \
           ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))
    
    return 1 - ssim.mean()


def psnr_loss(
    pred: torch.Tensor, 
    target: torch.Tensor, 
    max_val: Union[float, int] = 1.0,
    reduction: Literal['mean', 'sum', 'none'] = 'mean'
) -> torch.Tensor:
    """Computes PSNR (Peak Signal-to-Noise Ratio) between images.
    
    Args:
        pred: Predicted images tensor of shape (B, C, H, W)
        target: Target images tensor of shape (B, C, H, W)
        max_val: Maximum value of the signal (1.0 if image is normalized, 255 if not)
        reduction: 'mean', 'sum', or 'none' to return per image PSNR
        
    Returns:
        PSNR loss value (lower is worse). For optimization, often used as -PSNR
    """
    # Ensure inputs are float tensors
    pred = pred.float()
    target = target.float()
    
    # Convert max_val to tensor and move to same device as input
    max_val = torch.tensor(max_val, device=pred.device, dtype=torch.float)
    
    # Calculate MSE
    mse = torch.mean((pred - target) ** 2, dim=[1, 2, 3])
    
    # Avoid log of zero
    eps = torch.finfo(torch.float32).eps
    mse = torch.clamp(mse, min=eps)
    
    # Calculate PSNR
    psnr = 20 * torch.log10(max_val) - 10 * torch.log10(mse)
    
    # Apply reduction
    match reduction:
        case 'mean':
            return psnr.mean()
        case 'sum':
            return psnr.sum()
        case 'none':
            return psnr
        case _:
            raise ValueError(f"Invalid reduction method: {reduction}")


def combined_quality_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    alpha: float = 0.5,
    max_val: float = 1.0,
    normalize_psnr: bool = True,
    ssim_window_size: int = 16,
    ssim_sigma: float = 1.5,
    reduction: Literal['mean', 'sum', 'none'] = 'mean'
) -> torch.Tensor:
    """Combines SSIM and PSNR losses with weighted importance.
    
    Args:
        pred: Predicted images (B,C,H,W) in range [0,1]
        target: Target images (B,C,H,W) in range [0,1]
        alpha: Weight for SSIM loss (1-alpha will be PSNR weight)
        max_val: Maximum value of input tensors
        normalize_psnr: Whether to normalize PSNR to similar scale as SSIM
        ssim_window_size: Size of the gaussian window for SSIM
        ssim_sigma: Standard deviation of gaussian window for SSIM
        reduction: Reduction method for batch of images
        
    Returns:
        Combined loss value
    """
    # Calculate SSIM loss (ranges 0-1)
    ssim_loss_val = ssim_loss(pred, target, window_size=ssim_window_size, sigma=ssim_sigma)
    
    # Calculate PSNR loss
    psnr_val = psnr_loss(pred, target, max_val=max_val)
    
    if normalize_psnr:
        # Normalize PSNR to 0-1 range (assuming typical PSNR range of 0-50 dB)
        psnr_loss_val = 1.0 - (torch.clamp(psnr_val, 0, 50) / 50.0)
    else:
        # Use negative PSNR as loss
        psnr_loss_val = -psnr_val / 50.0  # Divide by 50 to balance magnitude
    
    # Combine losses
    combined_loss = alpha * ssim_loss_val + (1 - alpha) * psnr_loss_val
    
    # Apply reduction
    match reduction:
        case 'mean':
            return combined_loss.mean()
        case 'sum':
            return combined_loss.sum()
        case 'none':
            return combined_loss
        case _:
            raise ValueError(f"Invalid reduction method: {reduction}")
            

def kl_sparsity_loss(
    h: torch.Tensor, 
    target_sparsity: float = 0.05, 
    eps: float = 1e-6,
    clip_value: float = 20.0
) -> torch.Tensor:
    """Kullback-Leibler divergence based sparsity loss with enhanced numerical stability.
    
    Args:
        h: Input tensor of activations
        target_sparsity: Target activation rate (default: 0.05)
        eps: Small constant for numerical stability (default: 1e-6)
        clip_value: Maximum value to clip the loss (default: 20.0)
    
    Returns:
        torch.Tensor: KL divergence loss
    """
    # Ensure target_sparsity is valid
    target_sparsity = torch.clamp(torch.tensor(target_sparsity), eps, 1.0 - eps)
    
    # Average activation of each unit, with stronger clamping
    avg_activation = torch.clamp(torch.mean(h, dim=0), eps, 1.0 - eps)
    
    # Compute KL divergence with clipped values
    kl_div = target_sparsity * torch.log(target_sparsity / avg_activation) + \
             (1 - target_sparsity) * torch.log((1 - target_sparsity) / (1 - avg_activation))
    
    # Clip the loss to prevent extreme values
    kl_div = torch.clamp(kl_div, -clip_value, clip_value)
    
    return torch.sum(kl_div)

# Alternative version with dynamic weighting
def adaptive_quality_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    alpha_start: float = 0.8,
    alpha_end: float = 0.2,
    current_epoch: int = 0,
    total_epochs: int = 100
) -> torch.Tensor:
    """Combines SSIM and PSNR losses with dynamic weighting over training.
    
    Args:
        pred: Predicted images
        target: Target images
        alpha_start: Initial SSIM weight
        alpha_end: Final SSIM weight
        current_epoch: Current training epoch
        total_epochs: Total training epochs
        
    Returns:
        Combined loss value
    """
    # Calculate dynamic alpha
    alpha = alpha_start + (alpha_end - alpha_start) * (current_epoch / total_epochs)
    
    return combined_quality_loss(pred, target, alpha=alpha)       
 

def psnr_loss_for_training(
    pred: torch.Tensor, 
    target: torch.Tensor, 
    max_val: Union[float, int] = 1.0
) -> torch.Tensor:
    """Negative PSNR loss for training (since we want to maximize PSNR).
    
    Args:
        pred: Predicted images tensor of shape (B, C, H, W)
        target: Target images tensor of shape (B, C, H, W)
        max_val: Maximum value of the signal (1.0 if image is normalized, 255 if not)
        
    Returns:
        Negative PSNR value (for minimization during training)
    """
    return -psnr_loss(pred, target, max_val)    