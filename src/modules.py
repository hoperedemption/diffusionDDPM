import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

class ResidualBlock(nn.Module):
    """
    A standard residual block with two convolutional layers, GroupNorm, and GELU activation.
    """
    def __init__(self, in_channels: int, out_channels: int, time_embedding_dim: int):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.GroupNorm(32, in_channels),
            nn.GELU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        )
        self.time_proj = nn.Linear(time_embedding_dim, out_channels)
        self.conv2 = nn.Sequential(
            nn.GroupNorm(32, out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        )
        self.residual_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input feature map tensor.
            t_emb: Time embedding tensor.

        Returns:
            Output feature map tensor.
        """
        h = self.conv1(x)
        h += self.time_proj(t_emb).unsqueeze(-1).unsqueeze(-1) # Add time embedding
        h = self.conv2(h)
        return h + self.residual_conv(x)

class AttentionBlock(nn.Module):
    """
    Spatial self-attention block for U-Net feature maps.
    Uses multi-head self-attention followed by a feed-forward network.
    """
    def __init__(self, channels: int, num_heads: int = 4):
        """
        Args:
            channels: Number of input and output channels.
            num_heads: Number of attention heads.
        """
        super().__init__()
        self.norm = nn.GroupNorm(32, channels)
        self.attention = nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=num_heads,
            batch_first=True
        )
        self.ffn = nn.Sequential(
            nn.Linear(channels, channels * 4),
            nn.GELU(),
            nn.Linear(channels * 4, channels)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input feature map tensor of shape (B, C, H, W).

        Returns:
            Output feature map tensor of shape (B, C, H, W).
        """
        B, C, H, W = x.shape
        h = self.norm(x)
        h = h.view(B, C, H * W).transpose(1, 2)  # (B, H*W, C) for attention
        # H*W becomes the seq_len and C becomes the emb_dim
        
        attn_out, _ = self.attention(h, h, h)
        
        # Residual connection after attention
        h = attn_out + h
        
        # Feed-forward network
        ffn_out = self.ffn(h)
        # Residual connection after FFN
        h = ffn_out + h
        
        h = h.transpose(1, 2).view(B, C, H, W)  # (B, C, H, W)
        return h + x # Outer residual connection

class DownBlock(nn.Module):
    """
    Downsampling block for the U-Net.
    Consists of a ResidualBlock and an optional attention block, followed by a downsampling convolution.
    """
    def __init__(self, in_channels: int, out_channels: int, time_embedding_dim: int,
                 has_attention: bool = False, num_heads: int = 4):
        super().__init__()
        self.residual_block = ResidualBlock(in_channels, out_channels, time_embedding_dim)
        self.attention = AttentionBlock(out_channels, num_heads) if has_attention else nn.Identity()
        self.downsample = nn.Conv2d(out_channels, out_channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input feature map tensor.
            t_emb: Time embedding tensor.

        Returns:
            Downsampled and processed feature map tensor.
        """
        x = self.residual_block(x, t_emb)
        x = self.attention(x)
        x = self.downsample(x)
        return x

class UpBlock(nn.Module):
    """
    Upsampling block for the U-Net.
    Consists of an upsampling convolution, a concatenation with skip connection,
    a ResidualBlock, and an optional attention block.
    """
    def __init__(self, in_channels: int, out_channels: int, time_embedding_dim: int,
                 has_attention: bool = False, num_heads: int = 4):
        super().__init__()
        # in_channels for upsample conv refers to the channel dimension of x before concat.
        # it needs to be in_channels for the output of upsample conv  which matches the input to ResidualBlock after concat.
        self.upsample = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)
        # The residual block receives features from both upsample and skip connection.
        # so the input channels will be out_channels * 2.
        self.residual_block = ResidualBlock(out_channels * 2, out_channels, time_embedding_dim)
        self.attention = AttentionBlock(out_channels, num_heads) if has_attention else nn.Identity()

    def forward(self, x: torch.Tensor, skip: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input feature map tensor from the previous decoder stage.
            skip: Skip connection feature map tensor from the encoder.
            t_emb: Time embedding tensor.

        Returns:
            Upsampled and processed feature map tensor.
        """
        x = self.upsample(x)
        # Handle potential size mismatch after upsampling (e.g., due to odd dimensions)
        diff_h = skip.shape[2] - x.shape[2]
        diff_w = skip.shape[3] - x.shape[3]
        x = F.pad(x, [diff_w // 2, diff_w - diff_w // 2,
                      diff_h // 2, diff_h - diff_h // 2])

        x = torch.cat([x, skip], dim=1) # Concatenate along channels
        x = self.residual_block(x, t_emb)
        x = self.attention(x)
        return x

class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional embedding for timesteps.
    Based on the Transformer's positional encoding.
    """
    def __init__(self, embedding_dim: int):
        super().__init__()
        self.embedding_dim = embedding_dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: Timestep tensor of shape (B,).

        Returns:
            Positional embedding tensor of shape (B, embedding_dim).
        """
        device = t.device
        half_dim = self.embedding_dim // 2
        
        # compute the division term 10000^(2i/d_model)
        # exponent for base e is (torch.arange(0, half_dim) / half_dim) * -math.log(10000.0)
        
        # alternative, more numerically stable form for the division term:
        div_term = torch.exp(torch.arange(0, half_dim, device=device).float() * (-torch.log(torch.tensor(10000.0, device=device)) / half_dim))

        # Apply to timesteps
        # t needs to be unsqueezed to (B, 1) to broadcast with div_term (1, half_dim)
        t = t.unsqueeze(1).float()
        
        emb = t * div_term.unsqueeze(0) # (B, half_dim)
        
        # Concatenate sin and cos components
        emb = torch.cat([emb.sin(), emb.cos()], dim=1)
        
        return emb