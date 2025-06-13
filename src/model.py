import torch
import torch.nn as nn
from typing import Optional, Tuple
from modules import ResidualBlock, AttentionBlock, DownBlock, UpBlock, PositionalEncoding

class UNetConditional(nn.Module):
    """
    Conditional U-Net for denoising diffusion on images.
    This U-Net architecture takes in noisy images, timesteps, and optionally
    class labels, and predicts the noise added to the image.
    """
    def __init__(self,
                 in_channels: int = 3,
                 base_channels: int = 64,
                 time_embedding_dim: int = 256,
                 num_classes: Optional[int] = None,
                 attention_heads: int = 4):
        """
        Initializes the UNetConditional model.

        Args:
            in_channels: Number of input image channels (e.g., 3 for RGB).
            base_channels: Base number of channels for the first convolution layer.
                           Channel counts for subsequent layers scale up from this.
            time_embedding_dim: Dimension of the sinusoidal time embedding.
            num_classes: Number of classes in the dataset. If None, the model is unconditional.
            attention_heads: Number of attention heads for self-attention blocks.
        """
        super().__init__()
        self.num_classes = num_classes
        self.time_embedding_dim = time_embedding_dim

        # Time embedding
        self.time_encoder = nn.Sequential(
            PositionalEncoding(time_embedding_dim),
            nn.Linear(time_embedding_dim, time_embedding_dim),
            nn.GELU(),
            nn.Linear(time_embedding_dim, time_embedding_dim)
        )

        # Label embedding for conditioning
        if num_classes is not None:
            self.label_embedding = nn.Embedding(num_classes, time_embedding_dim)
            nn.init.xavier_uniform_(self.label_embedding.weight)

        # Initial convolution
        self.inc = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)

        # Encoder path
        # DownBlock: ResidualBlock -> Optional Attention -> Downsample
        self.down1 = DownBlock(base_channels, base_channels * 2, time_embedding_dim, has_attention=True, num_heads=attention_heads)
        self.down2 = DownBlock(base_channels * 2, base_channels * 4, time_embedding_dim, has_attention=True, num_heads=attention_heads)
        self.down3 = DownBlock(base_channels * 4, base_channels * 8, time_embedding_dim, has_attention=True, num_heads=attention_heads)

        # Bottleneck
        self.mid_block1 = ResidualBlock(base_channels * 8, base_channels * 8, time_embedding_dim)
        self.mid_attn = AttentionBlock(base_channels * 8, num_heads=attention_heads)
        self.mid_block2 = ResidualBlock(base_channels * 8, base_channels * 8, time_embedding_dim)

        # Decoder path
        # UpBlock: Upsample -> Concat Skip -> ResidualBlock -> Optional Attention
        # Note: in_channels for UpBlock refers to the channels after concatenation of x and skip.
        # The `upsample` conv_transpose outputs `out_channels`.
        self.up1 = UpBlock(base_channels * 8, base_channels * 4, time_embedding_dim, has_attention=True, num_heads=attention_heads)
        self.up2 = UpBlock(base_channels * 4, base_channels * 2, time_embedding_dim, has_attention=True, num_heads=attention_heads)
        self.up3 = UpBlock(base_channels * 2, base_channels, time_embedding_dim, has_attention=True, num_heads=attention_heads)

        # Output convolution
        self.outc = nn.Sequential(
            nn.GroupNorm(32, base_channels),
            nn.GELU(),
            nn.Conv2d(base_channels, in_channels, kernel_size=3, padding=1)
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor, y: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass of the U-Net.

        Args:
            x: Noisy image tensor of shape (B, C, H, W).
            t: Timestep tensor of shape (B,).
            y: Optional class label tensor of shape (B,) for conditional generation.
               If None, performs unconditional generation.

        Returns:
            Predicted noise tensor of the same shape as x.
        """
        # Time embedding
        t_emb = self.time_encoder(t)

        # Class conditioning
        if self.num_classes is not None and y is not None:
            # Randomly drop labels for classifier-free guidance
            # This is handled in the diffusion model's training loop where y is masked.
            label_emb = self.label_embedding(y)
            t_emb = t_emb + label_emb

        # Encoder
        x1 = self.inc(x) # Initial convolution
        x2 = self.down1(x1, t_emb) # Downsample 1
        x3 = self.down2(x2, t_emb) # Downsample 2
        x4 = self.down3(x3, t_emb) # Downsample 3

        # Bottleneck
        x_mid = self.mid_block1(x4, t_emb)
        x_mid = self.mid_attn(x_mid)
        x_mid = self.mid_block2(x_mid, t_emb)

        # Decoder (with skip connections)
        # skip_channels are the channel counts from the encoder path
        x = self.up1(x_mid, x3, t_emb) # Upsample 1, concat with x3
        x = self.up2(x, x2, t_emb)     # Upsample 2, concat with x2
        x = self.up3(x, x1, t_emb)     # Upsample 3, concat with x1

        # Output
        output = self.outc(x)
        return output