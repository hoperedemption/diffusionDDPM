import argparse
import os
import torch
from datetime import datetime

def parse_args():
    """
    Parses command-line arguments for the DDPM training script.

    Returns:
        argparse.Namespace: An object containing all the configuration arguments.
    """
    parser = argparse.ArgumentParser(description="Conditional Denoising Diffusion Probabilistic Model (DDPM)")

    # General
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility.")
    parser.add_argument("--output_dir", type=str, default="runs",
                        help="Directory to save logs, models, and samples.")
    parser.add_argument("--experiment_name", type=str,
                        default=f"ddpm_{datetime.now().strftime('%Y%m%d-%H%M%S')}",
                        help="Name of the current experiment for logging and checkpointing.")

    # Model
    parser.add_argument("--image_size", type=int, default=32,
                        help="Spatial size of the input images.")
    parser.add_argument("--in_channels", type=int, default=3,
                        help="Number of input channels for images (e.g., 3 for RGB).")
    parser.add_argument("--base_channels", type=int, default=64,
                        help="Base number of channels in the U-Net.")
    parser.add_argument("--time_embedding_dim", type=int, default=256,
                        help="Dimension of the time embedding.")
    parser.add_argument("--num_classes", type=int, default=10,
                        help="Number of classes in the dataset (e.g., 10 for CIFAR-10).")
    parser.add_argument("--attention_heads", type=int, default=4,
                        help="Number of attention heads in self-attention layers.")
    parser.add_argument("--unet_residual", type=bool, default=True,
                        help="Whether to use residual connections in DoubleConv blocks.")

    # Diffusion
    parser.add_argument("--timesteps", type=int, default=1000,
                        help="Number of diffusion timesteps (T).")
    parser.add_argument("--beta_schedule", type=str, default="linear", choices=["linear", "cosine"],
                        help="Type of beta schedule to use for diffusion.")
    parser.add_argument("--beta_start", type=float, default=0.0001,
                        help="Starting value of beta in the linear schedule.")
    parser.add_argument("--beta_end", type=float, default=0.02,
                        help="Ending value of beta in the linear schedule.")

    # Training
    parser.add_argument("--epochs", type=int, default=500,
                        help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=14,
                        help="Batch size for training.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="Number of batches to accumulate gradients over before updating model weights.")
    parser.add_argument("--learning_rate", type=float, default=2e-4,
                        help="Initial learning rate for the optimizer.")
    parser.add_argument("--optimizer", type=str, default="AdamW", choices=["Adam", "AdamW"],
                        help="Optimizer to use for training.")
    parser.add_argument("--ema_beta", type=float, default=0.9999,
                        help="Beta for Exponential Moving Average (EMA) of model weights.")
    parser.add_argument("--ema_warmup_steps", type=int, default=2000,
                        help="Number of steps before EMA starts smoothing heavily.")
    parser.add_argument("--amp", action="store_true",
                        help="Enable Automatic Mixed Precision (AMP) training (FP16).")
    parser.add_argument("--compile_model", action="store_true",
                        help="Enable torch.compile for model optimization.")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of worker processes for data loading.")
    parser.add_argument("--pin_memory", action="store_true",
                        help="Pin CUDA tensors in memory for faster data transfer.")
    parser.add_argument("--save_interval", type=int, default=50,
                        help="Save model checkpoint every N epochs.")
    parser.add_argument("--log_interval", type=int, default=10,
                        help="Log training metrics every N steps.")
    parser.add_argument("--sample_interval", type=int, default=1,
                        help="Generate and save sample images every N epochs.")
    parser.add_argument("--num_samples", type=int, default=64,
                        help="Number of images to sample during evaluation.")
    parser.add_argument("--guidance_scale", type=float, default=3.0,
                        help="Guidance scale for classifier-free guidance during sampling. Use 0 for unconditional.")
    parser.add_argument("--uncond_prob", type=float, default=0.1,
                        help="Probability of dropping class labels during training for classifier-free guidance.")

    args = parser.parse_args()

    # Create output directory and experiment-specific subdirectory
    args.log_dir = os.path.join(args.output_dir, args.experiment_name, "logs")
    args.checkpoint_dir = os.path.join(args.output_dir, args.experiment_name, "checkpoints")
    args.sample_dir = os.path.join(args.output_dir, args.experiment_name, "samples")
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.sample_dir, exist_ok=True)

    args.device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {args.device}")

    return args

if __name__ == '__main__':
    args = parse_args()
    print("--- Configuration ---")
    for arg, value in vars(args).items():
        print(f"{arg}: {value}")
    print("---------------------")