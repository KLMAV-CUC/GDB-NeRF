import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple


class SmoothLoss(nn.Module):
    def __init__(self,
                 loss_rate: List[float]) -> None:
        super(SmoothLoss, self).__init__()

        self.loss_rate = loss_rate
        self.num_stages = len(loss_rate)

    def forward(self, 
                depth_est_ms: List[torch.Tensor], 
                depth_tar_ms: List[torch.Tensor],
                mask_ms: List[torch.Tensor]) -> Tuple[torch.Tensor, dict]:
        """Multiscale L1 loss for depth estimation.

        Args:
            depth_est: (List[torch.Tensor]) shape [num_stages * (B, Hi, Wi)], estimted multiscale depth maps.
            depth_tar: (List[torch.Tensor]) shape [num_stages * (B, Hi, Wi)], target multiscale depth maps.
            mask: (List[torch.Tensor]) shape [num_stages * (B, Hi, Wi)], target multiscale masks.
        
        Returns:
            loss: (torch.Tensor)
        """
        loss = 0.0
        loss_dict = {}
        for i in range(self.num_stages):
            depth_est = depth_est_ms[i]
            depth_tar = depth_tar_ms[i]
            mask = mask_ms[i] > 0.5
            depth_loss = F.smooth_l1_loss(depth_est[mask], depth_tar[mask], reduction='mean')
            loss_dict[f'depth_loss{i}'] = depth_loss
            loss += self.loss_rate[i] * depth_loss

        return loss, loss_dict
