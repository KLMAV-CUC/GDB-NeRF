import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
from nerfacc.volrend import render_weight_from_alpha, accumulate_along_rays


def weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)


def render_weight_from_density(sigma: torch.Tensor, 
                               ray_indices: torch.Tensor, 
                               num_rays: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Transforms NeRF's predictions to semantically meaningful values.

    Args:
        sigma: (torch.Tensor) shape (num_samples,), density.
        bin_interval: (torch.Tensor) shape (num_samples,), depth interval of each sampled bin.
        ray_indices: (torch.Tensor) shape (num_samples,), ray indices of the samples.
        num_rays: (int) number of rays.

    Returns:
        weights: (torch.Tensor) shape (num_samples,), weights assigned to each sample.
        inverse_indices: (torch.Tensor) shape (num_samples,), relative ray indices of the samples.
    """
    alpha = 1. - torch.exp(-sigma)   # (num_samples,)
    weights, _ = render_weight_from_alpha(alpha, ray_indices=ray_indices, n_rays=num_rays)

    # Normalize weights
    _, inverse_indices = torch.unique_consecutive(ray_indices, return_inverse=True)
    weights_sum = torch.zeros(num_rays, device=sigma.device)
    weights_sum.index_add_(dim=0, index=inverse_indices, source=weights)
    weights = weights / weights_sum[inverse_indices].clamp_min(1e-6)
        
    return weights, inverse_indices


# def render_weight_from_density(sigma: torch.Tensor, 
#                                bin_interval: torch.Tensor, 
#                                ray_indices: torch.Tensor, 
#                                num_rays: int) -> torch.Tensor:
#     """Transforms NeRF's predictions to semantically meaningful values.

#     Args:
#         sigma: (torch.Tensor) shape (num_samples,), density.
#         bin_interval: (torch.Tensor) shape (num_samples,), depth interval of each sampled bin.
#         ray_indices: (torch.Tensor) shape (num_samples,), ray indices of the samples.
#         num_rays: (int) number of rays.

#     Returns:
#         weights: (torch.Tensor) shape (num_samples,), weights assigned to each sample.
#         inverse_indices: (torch.Tensor) shape (num_samples,), relative ray indices of the samples.
#     """
#     # Calculate alpha values
#     alpha = 1. - torch.exp(-sigma * bin_interval)  # (num_samples,)

#     # Get unique ray indices and counts of samples per ray
#     _, inverse_indices, samples_per_ray = torch.unique_consecutive(
#         ray_indices, return_inverse=True, return_counts=True
#     )

#     # Compute transmittance along the rays
#     device=sigma.device
#     max_samples_per_ray = samples_per_ray.max().item()
#     cumulative_counts = torch.cat([torch.tensor([0], device=device), samples_per_ray.cumsum(0)], dim=0)  # (num_rays+1,)
#     indices = torch.arange(len(alpha), device=device) - cumulative_counts[inverse_indices]  # (num_samples,)
#     T = torch.ones((num_rays, max_samples_per_ray+1), device=device)
#     T[inverse_indices, indices+1] = 1. - alpha + 1e-6
#     T = torch.cumprod(T[:, :-1], dim=-1)  # (num_rays, max_samples_per_ray)
    
#     # Compute normalized weights (PDF)
#     weights = alpha * T[inverse_indices, indices]
#     weights_sum = torch.zeros(num_rays, device=sigma.device)
#     weights_sum.index_add_(dim=0, index=inverse_indices, source=weights)
#     weights = weights / (weights_sum[inverse_indices] + 1e-6)

#     return weights, inverse_indices


def accumulate_value_along_rays(feat: torch.Tensor, 
                                z_vals: torch.Tensor, 
                                weights: torch.Tensor, 
                                ray_indices: torch.Tensor, 
                                num_rays: int, 
                                inverse_indices: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Transforms NeRF's predictions to semantically meaningful values.

    Args:
        feat: (torch.Tensor) shape (num_samples, channels), feature of samples.
        z_vals: (torch.Tensor) shape (num_samples,), depth/disparity value of samples.
        weights: (torch.Tensor) shape (num_samples,), weights assigned to each sample
        ray_indices: (torch.Tensor) shape (num_samples,), ray indices of the samples.
        num_rays: (int) number of rays.
        inverse_indices: (torch.Tensor) shape (num_samples,), relative ray indices of the samples.

    Returns:
        feat_map: (torch.Tensor) shape (num_rays, channels), estimated feature of rays.
        depth_map: (torch.Tensor) shape (num_rays,), estimated depth map of rays.
        opacity_map: (torch.Tensor) shape (num_rays,), estimated opacity map of rays.
    """
    values = torch.cat((feat, z_vals[:, None], torch.ones_like(z_vals[:, None])), dim=-1)  # (num_samples, channels+1+1)
    ray_values = accumulate_along_rays(weights, values, ray_indices, num_rays)  # (num_rays, channels+1+1)

    # if inverse_indices is None:
    #     _, inverse_indices = torch.unique_consecutive(ray_indices, return_inverse=True)  # (num_rays,)
    # ray_values = torch.zeros(num_rays, values.shape[-1], device=values.device)  # (num_rays, channels+1+1)
    # ray_values.index_add_(0, inverse_indices, values * weights.unsqueeze(-1))

    feat_map = ray_values[..., :-2]  # (num_rays, channels)
    depth_map = ray_values[..., -2]  # (num_rays,)
    opacity_map = ray_values[..., -1]  # (num_rays,)
        
    return feat_map, depth_map, opacity_map
