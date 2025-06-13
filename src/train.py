import torch
import torch.optim as optim
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import math
from torchvision.utils import save_image

from config import parse_args
from utils import set_seed, EMA, get_logger, TensorboardLogger, save_checkpoint, load_checkpoint, get_memory_info
from dataset import CIFAR10Dataset
from model import UNetConditional
from diffusion import GaussianDiffusion

def train():
    """
    Main training function for the Conditional DDPM.
    Orchestrates data loading, model training, logging, and checkpointing.
    """
    args = parse_args()

    # Set up logging
    logger = get_logger("ddpm_trainer", args.log_dir)
    tb_logger = TensorboardLogger(args.log_dir)
    logger.info(f"Starting experiment: {args.experiment_name}")
    logger.info(f"Configuration: {args}")

    # Set random seeds for reproducibility
    set_seed(args.seed)

    # Device configuration
    device = args.device
    logger.info(f"Using device: {device}")

    # Data loading
    cifar10_dataset = CIFAR10Dataset(image_size=args.image_size)
    train_dataloader = cifar10_dataset.get_train_dataloader(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory
    )
    logger.info(f"Loaded CIFAR-10 training data with {len(train_dataloader.dataset)} samples.")

    # Model and Diffusion
    model = UNetConditional(
        in_channels=args.in_channels,
        base_channels=args.base_channels,
        time_embedding_dim=args.time_embedding_dim,
        num_classes=args.num_classes,
        attention_heads=args.attention_heads
    ).to(device)

    diffusion = GaussianDiffusion(
        timesteps=args.timesteps,
        beta_schedule=args.beta_schedule,
        beta_start=args.beta_start,
        beta_end=args.beta_end,
        img_size=args.image_size,
        device=device
    )

    # Optimizer
    if args.optimizer == "AdamW":
        optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    else: # Default to Adam
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    # EMA
    ema = EMA(beta=args.ema_beta, warmup_steps=args.ema_warmup_steps)
    ema.register(model) # Register model parameters with EMA

    # Automatic Mixed Precision (AMP) Scaler
    scaler = GradScaler(enabled=args.amp)

    # torch.compile for performance
    if args.compile_model and torch.__version__ >= '2.0':
        logger.info("Compiling model with torch.compile...")
        try:
            # -->'inductor' is generally the recommended backend for speed.
            # -->'aot_eager' can be useful for debugging.
            model = torch.compile(model, mode="reduce-overhead") # -->'reduce-overhead' is good balance
            logger.info("Model compiled successfully.")
        except Exception as e:
            logger.error(f"Failed to compile model: {e}. Running without torch.compile.")

    # Load checkpoint if exists
    start_epoch = 0
    global_step = 0
    latest_checkpoint_path = os.path.join(args.checkpoint_dir, "latest_checkpoint.pt")
    if os.path.exists(latest_checkpoint_path):
        try:
            checkpoint_data = load_checkpoint(model, optimizer, ema, latest_checkpoint_path, device)
            start_epoch = checkpoint_data['epoch'] + 1
            global_step = checkpoint_data['step']
            logger.info(f"Resuming training from epoch {start_epoch}, step {global_step} with loaded loss {checkpoint_data['loss']:.4f}")
        except Exception as e:
            logger.error(f"Error loading checkpoint: {e}. Starting training from scratch.")
            # clear checkpoint data if loading fails to ensure a fresh start
            start_epoch = 0
            global_step = 0
    else:
        logger.info("No checkpoint found. Starting training from scratch.")

    # Training loop
    for epoch in range(start_epoch, args.epochs):
        model.train()
        pbar = tqdm(train_dataloader, desc=f"Epoch {epoch}/{args.epochs}", leave=False)
        total_loss = 0.0
        interval_loss = 0.0
        optimizer.zero_grad(set_to_none=True) # zero out grads before each epoch

        for batch_idx, (images, labels) in enumerate(pbar):
            images = images.to(device)
            labels = labels.to(device)

            # randomly select timesteps for each image in the batch
            t = torch.randint(0, args.timesteps, (images.shape[0],), device=device, dtype=torch.long)

            # Apply classifier-free guidance label dropout
            # Set a random percentage of labels to None for unconditional training
            if args.num_classes is not None and args.uncond_prob > 0:
                mask = torch.rand(labels.shape, device=device) < args.uncond_prob
                # Use a special index for unconditional or just pass None
                # Here, we pass None to the model if masked, which the model should handle.
                # The model expects `y` to be None for unconditional, not a special token.
                # So we create a `y_input` that is None when masked.
                y_input = labels.clone()
                y_input[mask] = args.num_classes # Use a value outside class range to indicate unconditional, or set to -1
                # The model's forward will interpret values outside valid class indices or None as unconditional.
                # For `nn.Embedding`, passing indices greater than or equal to `num_embeddings` will cause an error.
                # A safer approach for CFG training is to either pass `None` or a specific "unconditional" token
                # that `label_embedding` knows how to handle (e.g., by masking its contribution).
                # Let's adjust the diffusion.py to pass `None` if the mask is True.
                # For `diffusion.loss_fn`, we will pass `y` as is, and `diffusion.loss_fn` will handle the masking.
                # For now, let's keep `labels` as is and modify `diffusion.loss_fn` logic.
                # (Correction: `diffusion.py` currently takes `y` and `uncond_prob` to handle masking internally.
                # So, we pass `labels` directly.)
                pass # The handling is in diffusion.loss_fn now.

            with autocast(device_type=args.device.type, enabled=args.amp):
                # Calculate loss
                loss = diffusion.loss_fn(model, images, t, y=labels, uncond_prob=args.uncond_prob)
            
            # Scale loss and backpropagate
            scaler.scale(loss / args.gradient_accumulation_steps).backward()

            # Gradient accumulation
            if (batch_idx + 1) % args.gradient_accumulation_steps == 0 or (batch_idx + 1) == len(train_dataloader):
                scaler.unscale_(optimizer) # Unscale gradients before clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # Clip gradients to prevent exploding gradients
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True) # Clear gradients after update
                ema.update(model) # Update EMA weights after optimizer step

            total_loss += loss.item()
            interval_loss += loss.item()
            global_step += 1

            # Log metrics
            if global_step % args.log_interval == 0:
                avg_loss_this_interval = interval_loss / args.log_interval if (batch_idx + 1) % args.gradient_accumulation_steps == 0 else 0
                if avg_loss_this_interval > 0:
                    tb_logger.log_scalar("train/loss", avg_loss_this_interval, global_step)
                    mem_info = get_memory_info(device)
                    tb_logger.log_scalar("gpu_memory/allocated_mb", mem_info["allocated_mb"], global_step)
                    tb_logger.log_scalar("gpu_memory/cached_mb", mem_info["cached_mb"], global_step)

                    mem_info_str = f"{mem_info['allocated_mb']:.2f}MB / {mem_info['cached_mb']:.2f}MB (allocated/cached)"
                    pbar.set_postfix(loss=avg_loss_this_interval, step=global_step, gpu_mem=mem_info_str)
                    interval_loss = 0.0 # clear the interval loss for next epoch

        # End of epoch
        logger.info(f"Epoch {epoch} finished. Average loss: {total_loss / len(train_dataloader):.4f}")
        tb_logger.log_scalar("train/epoch_avg_loss", total_loss / len(train_dataloader), epoch)

        # Save checkpoint
        if (epoch + 1) % args.save_interval == 0 or (epoch + 1) == args.epochs:
            save_checkpoint(model, optimizer, ema, epoch, global_step, total_loss / len(train_dataloader),
                            args.checkpoint_dir, f"checkpoint_epoch_{epoch:03d}.pt")
            save_checkpoint(model, optimizer, ema, epoch, global_step, total_loss / len(train_dataloader),
                            args.checkpoint_dir, "latest_checkpoint.pt")
            logger.info(f"Saved checkpoint for epoch {epoch}.")

        # Generate and save samples
        if (epoch + 1) % args.sample_interval == 0 or (epoch + 1) == args.epochs:
            logger.info(f"Generating samples for epoch {epoch}...")
            model.eval() # Set model to evaluation mode
            
            # Use EMA model for sampling for better quality
            ema_model = UNetConditional(
                in_channels=args.in_channels,
                base_channels=args.base_channels,
                time_embedding_dim=args.time_embedding_dim,
                num_classes=args.num_classes,
                attention_heads=args.attention_heads
            ).to(device)
            ema.copy_to(ema_model) # Copy EMA weights to a separate model for sampling
            
            # Prepare dummy labels for conditional generation.
            # Generate `num_samples` images, distributed among classes.
            # E.g., if num_samples=64 and num_classes=10, generate 6-7 images per class.
            num_images_per_class = args.num_samples // args.num_classes
            remaining = args.num_samples % args.num_classes
            
            sample_labels = []
            for i in range(args.num_classes):
                sample_labels.extend([i] * num_images_per_class)
            sample_labels.extend(list(range(remaining)))
            
            sample_labels = torch.tensor(sample_labels, dtype=torch.long, device=device)
            # Shuffle labels to mix up samples
            sample_labels = sample_labels[torch.randperm(sample_labels.size(0))]
            
            # Ensure number of samples matches the length of generated labels
            sample_labels = sample_labels[:args.num_samples]


            with torch.no_grad():
                with autocast(device_type=args.device.type, enabled=args.amp):
                    # For sampling, always use the full number of classes for conditional guidance
                    # unless num_classes is None.
                    # The `diffusion.p_sample_loop` will handle the guidance scale.
                    if args.num_classes is not None:
                        generated_images = diffusion.p_sample_loop(
                            ema_model,
                            shape=(args.num_samples, args.in_channels, args.image_size, args.image_size),
                            y=sample_labels,
                            guidance_scale=args.guidance_scale
                        )
                    else:
                        generated_images = diffusion.p_sample_loop(
                            ema_model,
                            shape=(args.num_samples, args.in_channels, args.image_size, args.image_size),
                            y=None,
                            guidance_scale=0.0 # No guidance for unconditional model
                        )
            
            # Denormalize images from [-1, 1] to [0, 1] for saving
            generated_images = (generated_images + 1) / 2
            
            sample_filepath = os.path.join(args.sample_dir, f"samples_epoch_{epoch:03d}.png")
            save_image(generated_images, sample_filepath, nrow=int(math.sqrt(args.num_samples)))
            tb_logger.log_images("samples/generated_images", generated_images, global_step)
            logger.info(f"Saved {args.num_samples} samples to {sample_filepath}")
            del ema_model # Clear EMA model from memory
            torch.cuda.empty_cache() # Clear CUDA cache

    tb_logger.close()
    logger.info("Training complete.")
    # Log final hyperparameters and a dummy metric (e.g., final loss)
    tb_logger.log_hparams(vars(args), {"final_loss": total_loss / len(train_dataloader)})

if __name__ == "__main__":
    train()