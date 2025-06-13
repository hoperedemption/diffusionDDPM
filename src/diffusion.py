import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import math

class GaussianDiffusion:
    """
    Implements the forward and reverse processes of a Denoising Diffusion Probabilistic Model.
    Supports linear and cosine beta schedules.
    """
    def __init__(self,
                 timesteps: int = 1000,
                 beta_schedule: str = "linear",
                 beta_start: float = 0.0001,
                 beta_end: float = 0.02,
                 img_size: int = 32,
                 num_classes: int = 10,
                 device: torch.device = torch.device("cpu")):
        """
        Initializes the GaussianDiffusion model.

        Args:
            timesteps: Total number of diffusion steps (T).
            beta_schedule: Type of schedule for beta values ("linear" or "cosine").
            beta_start: Starting beta value for linear schedule.
            beta_end: Ending beta value for linear schedule.
            img_size: Size of the image (e.g., 32 for 32x32).
            num_classes: Number of classes in the dataset
            device: The device (CPU or CUDA) to perform computations on.
        """
        self.timesteps = timesteps
        self.beta_schedule = beta_schedule
        self.img_size = img_size
        self.device = device
        self.num_classes = num_classes

        # Define beta schedule
        self.betas = self._get_beta_schedule(beta_schedule, timesteps, beta_start, beta_end).to(device)

        # precompute values for the diffusion process for efficiency considerations
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0) # cum prod of all alphas
        # remove last value to shift alphas back by 1 -> preprend 1.0 at the start which corrrespond to t = 0
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = torch.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        self.sqrt_recip_alphas_minus_one = torch.sqrt(1.0 / self.alphas - 1)

        # Coefficients for reverse process (x_t to x_{t-1})
        # Calculates the variance of the posterior distribution p(x_{t-1} | x_t, x_0).
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        # Clamp posterior variance to a small value to avoid numerical issues
        # ensures numerical stability by preventing very small or zero variances.
        self.posterior_variance = torch.clamp(self.posterior_variance, min=1e-20)

        # calculates the log of the clamped posterior variance, used in sampling.
        self.posterior_log_variance_clipped = torch.log(self.posterior_variance.clamp(min=1e-20))

        # calculates the coefficients for the mean of the posterior distribution -> see diffusion formula
        self.posterior_mean_coef1 = (
            self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1.0 - self.alphas_cumprod)
        )

    def _get_beta_schedule(self, schedule_name: str, timesteps: int, beta_start: float, beta_end: float) -> torch.Tensor:
        """
        Generates beta values according to the specified schedule.

        Args:
            schedule_name: "linear" or "cosine".
            timesteps: Number of timesteps.
            beta_start: Starting value for beta.
            beta_end: Ending value for beta.

        Returns:
            A torch.Tensor of beta values.
        """
        if schedule_name == "linear":
            return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float32)
        elif schedule_name == "cosine":
            # From https://github.com/openai/guided-diffusion/blob/main/guided_diffusion/respace.py
            max_beta = 0.999
            f = lambda t: torch.cos(((t + 0.008) / 1.008) * math.pi / 2)**2
            betas = []
            for i in range(timesteps):
                t1 = i / timesteps
                t2 = (i + 1) / timesteps
                betas.append(min(1 - f(t2) / f(t1), max_beta))
            return torch.tensor(betas, dtype=torch.float32)
        else:
            raise NotImplementedError(f"Beta schedule '{schedule_name}' is not implemented.")

    def _extract(self, a: torch.Tensor, t: torch.Tensor, x_shape: Tuple[int, ...]) -> torch.Tensor:
        """
        Extracts values from a tensor a at specified timesteps t and
        reshapes them to match the input tensor x_shape.

        Args:
            a: Tensor from which to extract values (e.g., self.alphas_cumprod).
            t: Timestep tensor of shape (B,).
            x_shape: Shape of the target tensor (B, C, H, W) to reshape extracted values.

        Returns:
            Extracted and reshaped tensor.
        """
        batch_size = t.shape[0]
        out = a.gather(-1, t) # Ensure t is on CPU for gather index
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(self.device)

    def q_sample(self, x_start: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Applies noise to the input image `x_start` at timestep `t` (forward diffusion process).
        x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * epsilon

        Args:
            x_start: Original (clean) image tensor of shape (B, C, H, W).
            t: Timestep tensor of shape (B,).
            noise: Optional noise tensor of the same shape as x_start. If None,
                   random noise is generated.

        Returns:
            Noisy image tensor x_t.
        """
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alphas_cumprod_t = self._extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)

        x_t = sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
        return x_t

    def p_mean_variance(self, model: nn.Module, x: torch.Tensor, t: torch.Tensor,
                        y: Optional[torch.Tensor] = None, guidance_scale: float = 0.0) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Predicts the mean and variance of p(x_{t-1} | x_t) using the Denoising U-Net.
        Applies classifier-free guidance if y and guidance_scale > 0 are provided.

        Args:
            model: The U-Net model that predicts noise.
            x: Noisy image tensor x_t of shape (B, C, H, W).
            t: Current timestep tensor of shape (B,).
            y: Optional class label tensor of shape (B,) for conditional generation.
            guidance_scale: Strength of classifier-free guidance. If 0, no guidance.

        Returns:
            A tuple containing:
            - model_mean: Predicted mean of x_{t-1}.
            - posterior_variance: Variance of x_{t-1}.
            - model_log_variance: Log variance of x_{t-1}.
        """
        if guidance_scale > 0.0 and y is not None:
            # Predict noise for conditional (y) and unconditional (None)
            # This requires running the model twice or cleverly combining
            # batch for single run.
            
            # For simplicity, running twice here. 
            
            # Unconditional prediction (y=None)
            uncond_pred_noise = model(x, t, y=None) # y=None implies unconditional
            
            # Conditional prediction
            cond_pred_noise = model(x, t, y=y)
            
            # Classifier-free guidance -> combines unconditional and conditional noise predictions.
            pred_noise = uncond_pred_noise + guidance_scale * (cond_pred_noise - uncond_pred_noise)
        else:
            pred_noise = model(x, t, y=y)

        # Coefficients for predicting x_0 from x_t and predicted noise
        # x_0_pred = (x_t - sqrt(1-alpha_bar_t) * pred_noise) / sqrt(alpha_bar_t)
        sqrt_recip_alphas_t = self._extract(self.sqrt_recip_alphas, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x.shape)
        alphas_t = self._extract(self.alphas, t, x.shape)
        
        x_0_pred = sqrt_recip_alphas_t * (x - sqrt_one_minus_alphas_cumprod_t * pred_noise)
        x_0_pred = torch.clamp(x_0_pred, -1.0, 1.0) # Clamp predicted x_0 to original image range

        # Compute mean and variance for p(x_{t-1} | x_t, x_0_pred)
        # Posterior mean = coef1 * x_0_pred + coef2 * x_t
        posterior_mean_coef1_t = self._extract(self.posterior_mean_coef1, t, x.shape)
        posterior_mean_coef2_t = self._extract(self.posterior_mean_coef2, t, x.shape)
        
        model_mean = posterior_mean_coef1_t * x_0_pred + posterior_mean_coef2_t * x
        
        # Posterior variance (fixed by definition of DDPM)
        posterior_variance_t = self._extract(self.posterior_variance, t, x.shape)
        posterior_log_variance_clipped_t = self._extract(self.posterior_log_variance_clipped, t, x.shape)

        # returns the predicted mean, variance, and log variance for the reverse step.
        return model_mean, posterior_variance_t, posterior_log_variance_clipped_t

    @torch.no_grad()
    def p_sample(self, model: nn.Module, x: torch.Tensor, t: int,
                 y: Optional[torch.Tensor] = None, guidance_scale: float = 0.0) -> torch.Tensor:
        """
        Samples x_{t-1} from x_t (one step of reverse diffusion).

        Args:
            model: The U-Net model that predicts noise.
            x: Noisy image tensor x_t of shape (B, C, H, W).
            t: Current timestep (integer, not tensor).
            y: Optional class label tensor for conditional generation.
            guidance_scale: Strength of classifier-free guidance.

        Returns:
            Denoised image tensor x_{t-1}.
        """
        # Convert t to a tensor for extraction, ensuring it's on the correct device
        t_tensor = torch.full((x.shape[0],), t, device=self.device, dtype=torch.long)

        model_mean, _, model_log_variance = self.p_mean_variance(model, x, t_tensor, y, guidance_scale)

        # Generate noise if t > 0
        noise = torch.randn_like(x)
        if t == 0:
            return model_mean
        else:
            return model_mean + torch.exp(0.5 * model_log_variance) * noise

    @torch.no_grad()
    def p_sample_loop(self, model: nn.Module, shape: Tuple[int, ...],
                      y: Optional[torch.Tensor] = None, guidance_scale: float = 0.0) -> torch.Tensor:
        """
        Generates images by iteratively denoising from pure noise (reverse diffusion process).

        Args:
            model: The U-Net model that predicts noise.
            shape: Desired shape of the output images (B, C, H, W).
            y: Optional class label tensor for conditional generation.
            guidance_scale: Strength of classifier-free guidance.

        Returns:
            Generated image tensor.
        """
        # Start with pure noise
        img = torch.randn(shape, device=self.device)

        # Iterate backward through timesteps
        for i in reversed(range(0, self.timesteps)):
            img = self.p_sample(model, img, i, y, guidance_scale)
            # i could explicitly del intermediate tensors if memory becomes an issue -> free the memory used
            # del t_tensor, model_mean, model_log_variance, noise 

        # clamp and normalize images to [0, 1] for visualization if needed,
        # otherwise keep them in [-1, 1] range as per training.
        # here, we keep them in [-1, 1] as they are model outputs.
        return img

    def loss_fn(self, model: nn.Module, x_0: torch.Tensor, t: torch.Tensor,
                y: Optional[torch.Tensor] = None, uncond_prob: float = 0.1) -> torch.Tensor:
        """
        Calculates the training loss (L_simple, as in the paper, here MSE between true and predicted noise).

        Args:
            model: The U-Net model that predicts noise.
            x_0: Clean image tensor of shape (B, C, H, W).
            t: Timestep tensor of shape (B,).
            y: Optional class label tensor for conditional generation.
            uncond_prob: Probability of setting y to None for classifier-free guidance training.

        Returns:
            The calculated loss tensor.
        """
        noise = torch.randn_like(x_0)
        x_t = self.q_sample(x_0, t, noise)

        # Apply classifier-free guidance masking during training
        if y is not None and uncond_prob > 0:
            mask = torch.rand(y.shape, device=y.device) < uncond_prob
            y = torch.where(mask, torch.full_like(y, fill_value=self.num_classes, dtype=torch.long), y) # assuming num_classes is a special token, or just pass None

        predicted_noise = model(x_t, t, y)
        loss = F.mse_loss(noise, predicted_noise)
        return loss