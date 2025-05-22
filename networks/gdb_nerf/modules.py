import torch.nn as nn
from typing import Union, Tuple


def conv_block2d(in_channels: int, 
                 out_channels: int, 
                 kernel_size: Union[int, Tuple[int, int]], 
                 stride: Union[int, Tuple[int, int]] = (1, 1), 
                 padding: Union[int, Tuple[int, int]] = (0, 0), 
                 groups: int = 1) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=False), 
        nn.BatchNorm2d(out_channels), 
        # nn.GroupNorm(num_groups=1, num_channels=out_channels), 
        nn.ReLU(inplace=True)
    )


def deconv_block2d(in_channels: int, 
                   out_channels: int, 
                   kernel_size: Union[int, Tuple[int, int]], 
                   stride: Union[int, Tuple[int, int]] = (1, 1), 
                   padding: Union[int, Tuple[int, int]] = (0, 0), 
                   output_padding: Union[int, Tuple[int, int]] = (0, 0)) -> nn.Sequential:
    return nn.Sequential(
        nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding, bias=False), 
        nn.BatchNorm2d(out_channels), 
        # nn.GroupNorm(num_groups=1, num_channels=out_channels), 
        nn.ReLU(inplace=True)
    )


def conv_block3d(in_channels: int, 
                 out_channels: int, 
                 kernel_size: Union[int, Tuple[int, int, int]], 
                 stride: Union[int, Tuple[int, int, int]] = (1, 1, 1), 
                 padding: Union[int, Tuple[int, int, int]] = (0, 0, 0)) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding, bias=False), 
        nn.BatchNorm3d(out_channels), 
        # nn.GroupNorm(num_groups=1, num_channels=out_channels), 
        nn.ReLU(inplace=True)
    )


def deconv_block3d(in_channels: int, 
                   out_channels: int, 
                   kernel_size: Union[int, Tuple[int, int, int]], 
                   stride: Union[int, Tuple[int, int, int]] = (1, 1, 1), 
                   padding: Union[int, Tuple[int, int, int]] = (0, 0, 0), 
                   output_padding: Union[int, Tuple[int, int, int]] = (0, 0, 0)) -> nn.Sequential:
    return nn.Sequential(
        nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride, padding, output_padding, bias=False), 
        nn.BatchNorm3d(out_channels), 
        # nn.GroupNorm(num_groups=1, num_channels=out_channels), 
        nn.ReLU(inplace=True)
    )
