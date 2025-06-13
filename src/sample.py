import torch
import os
from torchvision.utils import save_image
from tqdm import tqdm
import math

from config import parse_args
from utils import set_seed, EMA, load_checkpoint
from model import UNetConditional
from diffusion import GaussianDiffusion

def sample_images():
    """
    Script to load a trained DDPM model and generate new images.
    """
    args = parse_args()

    # Device configuration
    device = args.device
    print(f"Using device: {device}")

    # Set random seed
    set_seed(args.seed)

    # Initialize model
    model = UNetConditional(
        in_channels=args.in_channels,
        base_channels=args.base_channels,
        time_embedding_dim=args.time_embedding_dim,
        num_classes=args.num_classes,
        attention_heads=args.attention_heads
    ).to(device)

    # Initialize EMA (even if not loading EMA state, it's needed for the structure)
    ema = EMA(beta=args.ema_beta, warmup_steps=args.ema_warmup_steps)
    
    # Load checkpoint
    # You might want to specify the exact checkpoint path or always load the latest.
    # For sampling, typically load the EMA model weights.
    checkpoint_path = os.path.join(args.checkpoint_dir, "latest_checkpoint.pt")
    # You might want to load a specific epoch checkpoint if needed
    # checkpoint_path = os.path.join(args.checkpoint_dir, "checkpoint_epoch_XXX.pt")

    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint not found at {checkpoint_path}. Please train a model first.")
        return

    try:
        # Pass a dummy optimizer if not loading its state, or None.
        # Here we don't need optimizer state for sampling.
        load_checkpoint(model, optimizer=None, ema=ema, checkpoint_path=checkpoint_path, device=device)
        print("Model and EMA weights loaded successfully.")
    except Exception as e:
        print(f"Failed to load model from checkpoint: {e}")
        return

    # Copy EMA weights to the model for inference
    ema.copy_to(model)
    model.eval() # Set model to evaluation mode

    # Initialize Diffusion
    diffusion = GaussianDiffusion(
        timesteps=args.timesteps,
        beta_schedule=args.beta_schedule,
        beta_start=args.beta_start,
        beta_end=args.beta_end,
        img_size=args.image_size,
        device=device
    )

    print(f"Generating {args.num_samples} images...")
    
    # Prepare labels for conditional generation
    if args.num_classes is not None:
        num_images_per_class = args.num_samples // args.num_classes
        remaining = args.num_samples % args.num_classes
        
        sample_labels = []
        for i in range(args.num_classes):
            sample_labels.extend([i] * num_images_per_class)
        sample_labels.extend(list(range(remaining)))
        
        sample_labels = torch.tensor(sample_labels, dtype=torch.long, device=device)
        sample_labels = sample_labels[torch.randperm(sample_labels.size(0))]
        sample_labels = sample_labels[:args.num_samples]
    else:
        sample_labels = None # Unconditional generation

    with torch.no_grad():
        # Perform reverse diffusion to generate images
        generated_images = diffusion.p_sample_loop(
            model,
            shape=(args.num_samples, args.in_channels, args.image_size, args.image_size),
            y=sample_labels,
            guidance_scale=args.guidance_scale
        )
    
    # Denormalize images from [-1, 1] to [0, 1] for saving
    generated_images = (generated_images + 1) / 2

    # Save generated images
    output_sample_dir = os.path.join(args.output_dir, args.experiment_name, "generated_samples")
    os.makedirs(output_sample_dir, exist_ok=True)
    sample_filepath = os.path.join(output_sample_dir, f"generated_{args.experiment_name}_{args.num_samples}_samples_cfg_{args.guidance_scale}.png")
    
    # Save as a grid for better visualization
    save_image(generated_images, sample_filepath, nrow=int(math.sqrt(args.num_samples)))
    print(f"Generated {args.num_samples} samples and saved to {sample_filepath}")

if __name__ == "__main__":
    sample_images()