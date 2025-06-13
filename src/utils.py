import torch
import torch.nn as nn
import numpy as np
import random
import os
import logging
from typing import Optional, Dict, Any
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

def set_seed(seed: int) -> None:
    """
    Sets the random seed for reproducibility across different libraries.

    Args:
        seed: The integer seed to use.
    """
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False # Set to False for reproducibility, True for potential speed.
    np.random.seed(seed)
    random.seed(seed)
    print(f"Random seed set to {seed}")

class EMA:
    """
    Exponential Moving Average (EMA) handler for model parameters.
    Smooths training by tracking a momentum-based average of weights.
    """
    def __init__(self, beta: float = 0.9999, warmup_steps: int = 2000):
        """
        Initializes the EMA handler.

        Args:
            beta: Momentum decay factor (0 < beta < 1). Higher beta means more smoothing.
            warmup_steps: Number of initial updates during which EMA model directly
                          copies the training model's weights.
        """
        self.beta = beta
        self.warmup_steps = warmup_steps
        self.updates = 0
        self.shadow_params = [] # Stores EMA weights

    def register(self, model: nn.Module) -> None:
        """
        Registers the model for EMA tracking. Must be called after model creation.

        Args:
            model: The PyTorch model to track.
        """
        # shadow params are accumulated by the EMA over all update steps
        self.shadow_params = [p.clone().detach() for p in model.parameters() if p.requires_grad]

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        """
        Updates EMA model parameters from the current training model.

        Args:
            model: Current training model.
        """
        if not self.shadow_params:
            # First update, initialize shadow parameters
            self.register(model)
        
        self.updates += 1
        # the ema weights are first increased slowly (1/10, 2/11) - EMA is not heavily updated 
        # the primary weights are rapidly changing
        # as self.updates -> +inf this becomes ratio becomes 1 and thus reaches its peak value which is self.beta
        decay = min(self.beta, (1 + self.updates) / (10 + self.updates)) # Adaptive decay, common in diffusion
        
        for shadow_p, model_p in zip(self.shadow_params, model.parameters()):
            if model_p.requires_grad:
                if self.updates <= self.warmup_steps:
                    shadow_p.copy_(model_p.data)
                else:
                    shadow_p.mul_(decay).add_(model_p.data, alpha=1.0 - decay)

    @torch.no_grad()
    def copy_to(self, model: nn.Module) -> None:
        """
        Copies the EMA parameters to the provided model.
        Useful for evaluation or sampling.

        Args:
            model: The PyTorch model to copy EMA weights to.
        """
        for shadow_p, model_p in zip(self.shadow_params, model.parameters()):
            if model_p.requires_grad:
                model_p.copy_(shadow_p)

    def state_dict(self) -> Dict[str, Any]:
        """Returns the state dictionary of the EMA."""
        return {
            'beta': self.beta,
            'warmup_steps': self.warmup_steps,
            'updates': self.updates,
            'shadow_params': self.shadow_params
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Loads the state dictionary into the EMA."""
        self.beta = state_dict['beta']
        self.warmup_steps = state_dict['warmup_steps']
        self.updates = state_dict['updates']
        self.shadow_params = state_dict['shadow_params']


def get_logger(name: str, log_dir: str) -> logging.Logger:
    """
    Configures and returns a logger with console and file handlers.

    Args:
        name: Name of the logger.
        log_dir: Directory to save log files.

    Returns:
        A configured logging.Logger instance.
    """
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"{name}_{datetime.now().strftime('%Y%m%d-%H%M%S')}.log")

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False # Prevent messages from being passed to the root logger

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # File handler
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger

class TensorboardLogger:
    """
    Wrapper for TensorBoard SummaryWriter for unified logging.
    """
    def __init__(self, log_dir: str):
        """
        Initializes the TensorboardLogger.

        Args:
            log_dir: Directory where TensorBoard logs will be stored.
        """
        self.writer = SummaryWriter(log_dir=log_dir)

    def log_scalar(self, tag: str, value: float, step: int) -> None:
        """Logs a scalar value to TensorBoard."""
        self.writer.add_scalar(tag, value, step)

    def log_images(self, tag: str, images: torch.Tensor, step: int) -> None:
        """Logs image tensors to TensorBoard."""
        # ensure images are in [0, 1] range and correct format (N, C, H, W)
        if images.dtype == torch.uint8:
            images = images.float() / 255.0
        elif images.max() > 1.0 or images.min() < 0.0:
            images = (images.clamp(-1, 1) + 1) / 2.0 # min max norm here assumes images are scaled to [-1, 1]

        self.writer.add_images(tag, images, step)

    def log_hparams(self, hparam_dict: Dict[str, Any], metric_dict: Dict[str, float]) -> None:
        """Logs hyperparameters and final metrics."""
        self.writer.add_hparams(hparam_dict, metric_dict)

    def close(self) -> None:
        """Closes the TensorBoard writer."""
        self.writer.close()

def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    ema: Optional[EMA],
    epoch: int,
    step: int,
    loss: float,
    checkpoint_dir: str,
    filename: str = "checkpoint.pt"
) -> None:
    """
    Saves the model, optimizer, EMA state, and training progress.

    Args:
        model: The PyTorch model to save.
        optimizer: The optimizer to save.
        ema: The EMA object to save its state (optional).
        epoch: Current training epoch.
        step: Current training step.
        loss: Current training loss.
        checkpoint_dir: Directory to save the checkpoint.
        filename: Name of the checkpoint file.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    filepath = os.path.join(checkpoint_dir, filename)
    state = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'step': step,
        'loss': loss,
        'timestamp': datetime.now().strftime('%Y%m%d-%H%M%S')
    }
    if ema is not None:
        state['ema_state_dict'] = ema.state_dict()

    torch.save(state, filepath)
    print(f"Checkpoint saved to {filepath}")

def load_checkpoint(
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    ema: Optional[EMA],
    checkpoint_path: str,
    device: torch.device,
    strict: bool = True
) -> Dict[str, Any]:
    """
    Loads a checkpoint into the model, optimizer, and EMA.

    Args:
        model: The PyTorch model to load state into.
        optimizer: The optimizer to load state into (optional).
        ema: The EMA object to load state into (optional).
        checkpoint_path: Path to the checkpoint file.
        device: The device to load the checkpoint onto.
        strict: Whether to strictly enforce that the keys in state_dict match the keys returned by this module's state_dict().

    Returns:
        A dictionary containing loaded training progress (epoch, step, loss).
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found at {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'], strict=strict)
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if ema is not None and 'ema_state_dict' in checkpoint:
        ema.load_state_dict(checkpoint['ema_state_dict'])

    print(f"Checkpoint loaded from {checkpoint_path}")
    return {
        'epoch': checkpoint.get('epoch', 0),
        'step': checkpoint.get('step', 0),
        'loss': checkpoint.get('loss', float('inf'))
    }

def get_memory_info(device: torch.device) -> Dict[str, float]:
    """
    Retrieves current GPU memory usage (if CUDA is available).

    Args:
        device: The device to check memory for.

    Returns:
        A dictionary with memory usage in MB.
    """
    if device.type == 'cuda':
        allocated = torch.cuda.memory_allocated(device) / (1024 ** 2)
        cached = torch.cuda.memory_reserved(device) / (1024 ** 2)
        return {"allocated_mb": allocated, "cached_mb": cached}
    elif device.type == 'mps':
        # These values are for macOS MPS backend
        allocated = torch.mps.current_allocated_memory() / (1024 ** 2)
        cached = torch.mps.driver_allocated_memory() / (1024 ** 2)
        return {"allocated_mb": allocated, "cached_mb": cached}
    return {"allocated_mb": 0.0, "cached_mb": 0.0}

def print_cuda_memory_summary(device: torch.device) -> None:
    """
    Prints a detailed CUDA memory summary.
    """
    if device.type == 'cuda':
        print("\n--- CUDA Memory Summary ---")
        print(torch.cuda.memory_summary(device=device, abbreviated=True))
        print("---------------------------\n")
        