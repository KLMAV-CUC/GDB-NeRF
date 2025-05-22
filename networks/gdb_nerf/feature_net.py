import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List
from .modules import conv_block2d


class FeatureNet(nn.Module):
    """Feature pyramid network for feature extraction from 2D images
    """

    def __init__(self, 
                 base_channels: int = 8, 
                 out_channels: Tuple[int, int, int] = (32, 16, 8)) -> None:
        super(FeatureNet, self).__init__()

        self.conv0 = nn.Sequential(
            conv_block2d(3, base_channels, kernel_size=3, padding=1),
            conv_block2d(base_channels, base_channels, kernel_size=3, padding=1)
        )

        self.conv1 = nn.Sequential(
            conv_block2d(base_channels, base_channels * 2, kernel_size=5, stride=2, padding=2),
            conv_block2d(base_channels * 2, base_channels * 2, kernel_size=3, padding=1)
        )

        self.conv2 = nn.Sequential(
            conv_block2d(base_channels * 2, base_channels * 4, kernel_size=5, stride=2, padding=2),
            conv_block2d(base_channels * 4, base_channels * 4, kernel_size=3, padding=1)
        )

        self.out0 = nn.Conv2d(base_channels * 4, out_channels[0], kernel_size=1)

        final_chs = base_channels * 4
        self.inner1 = nn.Conv2d(base_channels * 2, final_chs, kernel_size=1)
        self.inner2 = nn.Conv2d(base_channels, final_chs, kernel_size=1)
        self.out1 = nn.Conv2d(final_chs, out_channels[1], kernel_size=3, padding=1, bias=False)
        self.out2 = nn.Conv2d(final_chs, out_channels[2], kernel_size=3, padding=1, bias=False)

    def forward(self, 
                x: torch.Tensor) -> List[torch.Tensor]:
        """ Extrat two-stage pyramid features from 2D images.
        
        Args:
            x (torch.Tensor): shape (B, 3, H, W).
        
        Returns:
            feats (List[torch.Tensor]): features of each level from coarsest to finest.
        """
        conv0 = self.conv0(x)  # (B, base_channels, H, W)
        conv1 = self.conv1(conv0)  # (B, base_channels * 2, H/2, W/2)
        conv2 = self.conv2(conv1)  # (B, base_channels * 4, H/4, W/4)
        
        feats = []
        intra_feat = conv2
        feats.append(self.out0(intra_feat))  # (B, out_channels[0], H/4, W/4)

        intra_feat = F.interpolate(intra_feat, size=conv1.shape[-2:], mode='nearest') + self.inner1(conv1)
        feats.append(self.out1(intra_feat))  # (B, out_channels[1], H/2, W/2)

        intra_feat = F.interpolate(intra_feat, size=conv0.shape[-2:], mode='nearest') + self.inner2(conv0)
        feats.append(self.out2(intra_feat))  # (B, out_channels[2], H, W)

        return feats
