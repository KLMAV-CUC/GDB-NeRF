import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
from .modules import conv_block3d, deconv_block3d
    

class CostRegNet(nn.Module):
    def __init__(self, 
                 in_channels: int, 
                 out_channels: int, 
                 base_channels: int) -> None:
        """3D Unet for cost volume regularization.
        
        Args:
            in_channels (int): input channels.
            out_channels (int): output channels.
            base_channels (int): base channels.
        """
        super(CostRegNet, self).__init__()

        self.conv0 = conv_block3d(in_channels, base_channels, kernel_size=3, padding=1)

        self.conv1 = conv_block3d(base_channels, base_channels * 2, kernel_size=3, stride=2, padding=1)
        self.conv2 = conv_block3d(base_channels * 2, base_channels * 2, kernel_size=3, padding=1)

        self.conv3 = conv_block3d(base_channels * 2, base_channels * 4, kernel_size=3, stride=2, padding=1)
        self.conv4 = conv_block3d(base_channels * 4, base_channels * 4, kernel_size=3, padding=1)

        self.conv5 = conv_block3d(base_channels * 4, base_channels * 8, kernel_size=3, stride=2, padding=1)
        self.conv6 = conv_block3d(base_channels * 8, base_channels * 8, kernel_size=3, padding=1)

        self.conv7 = deconv_block3d(base_channels * 8, base_channels * 4, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv8 = deconv_block3d(base_channels * 4, base_channels * 2, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv9 = deconv_block3d(base_channels * 2, base_channels, kernel_size=3, stride=2, padding=1, output_padding=1)

        self.feat_head = nn.Conv3d(base_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.prob_head = nn.Conv3d(base_channels, 1, kernel_size=3, padding=1, bias=False)

    def forward(self, 
                x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply cost volume regularization.
        
        Args:
            x (torch.Tensor): shape (B, in_channels, D, H, W), input volume.
        
        Returns:
            feat (torch.Tensor): shape (B, out_channels, D, H, W), feature volume.
            depth_prob (torch.Tensor): shape (B, D, H, W), depth probability volume.
        """
        
        conv0 = self.conv0(x)
        conv2 = self.conv2(self.conv1(conv0))
        conv4 = self.conv4(self.conv3(conv2))
        x = self.conv6(self.conv5(conv4))
        
        x = conv4 + self.conv7(x)
        x = conv2 + self.conv8(x)
        x = conv0 + self.conv9(x)

        feat = self.feat_head(x)
        depth_prob = self.prob_head(x).squeeze(1)  # (B, D, H, W)
        depth_prob = F.softmax(depth_prob, dim=1)

        return feat, depth_prob
    

class CostRegNet_small(nn.Module):
    def __init__(self, 
                 in_channels: int, 
                 out_channels: int, 
                 base_channels: int) -> None:
        """Small 3D Unet for cost volume regularization.
        
        Args:
            in_channels (int): input channels.
            out_channels (int): output channels.
            base_channels (int): base channels.
        """
        super(CostRegNet_small, self).__init__()

        self.conv0 = conv_block3d(in_channels, base_channels, kernel_size=3, padding=1)

        self.conv1 = conv_block3d(base_channels, base_channels * 2, kernel_size=3, stride=2, padding=1)
        self.conv2 = conv_block3d(base_channels * 2, base_channels * 2, kernel_size=3, padding=1)

        self.conv3 = conv_block3d(base_channels * 2, base_channels * 4, kernel_size=3, stride=2, padding=1)
        self.conv4 = conv_block3d(base_channels * 4, base_channels * 4, kernel_size=3, padding=1)

        self.conv5 = deconv_block3d(base_channels * 4, base_channels * 2, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv6 = deconv_block3d(base_channels * 2, base_channels, kernel_size=3, stride=2, padding=1, output_padding=1)
        
        self.feat_head = nn.Conv3d(base_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.prob_head = nn.Conv3d(base_channels, 1, kernel_size=3, padding=1, bias=False)

    def forward(self, 
                x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply cost volume regularization.
        
        Args:
            x (torch.Tensor): shape (B, in_channels, D, H, W), input volume.
        
        Returns:
            feat (torch.Tensor): shape (B, out_channels, D, H, W), feature volume.
            depth_prob (torch.Tensor): shape (B, D, H, W), depth probability volume.
        """
        conv0 = self.conv0(x)
        conv2 = self.conv2(self.conv1(conv0))
        x = self.conv4(self.conv3(conv2))

        x = conv2 + self.conv5(x)
        x = conv0 + self.conv6(x)

        feat = self.feat_head(x)
        depth_prob = self.prob_head(x).squeeze(1)  # (B, D, H, W)
        depth_prob = F.softmax(depth_prob, dim=1)

        return feat, depth_prob