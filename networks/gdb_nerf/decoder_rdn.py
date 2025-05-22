import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SEBlock2D(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEBlock2D, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c = x.shape[:2]
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class ResidualDenseBlock(nn.Module):
    def __init__(self, 
                 num_feats: int, 
                 growth_rate: int = 32) -> None:
        super(ResidualDenseBlock, self).__init__()
        self.conv1 = nn.Conv2d(num_feats, growth_rate, kernel_size=3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(num_feats + growth_rate, growth_rate, kernel_size=3, padding=1, bias=False)
        self.conv3 = nn.Conv2d(num_feats + 2 * growth_rate, num_feats, kernel_size=3, padding=1, bias=False)
        self.se = SEBlock2D(num_feats)

    def forward(self, 
                x: torch.Tensor) -> torch.Tensor:
        x1 = F.relu(self.conv1(x), inplace=True)
        x2 = F.relu(self.conv2(torch.cat([x, x1], dim=1)), inplace=True)
        x3 = self.conv3(torch.cat([x, x1, x2], dim=1))
        x = x + self.se(x3)
        return x


class Decoder(nn.Module):
    def __init__(self, 
                 in_channels: int, 
                 out_channels: int, 
                 num_feats: int, 
                 num_layers: int, 
                 upscale_factor: int) -> None: 
        super(Decoder, self).__init__()
        if upscale_factor <= 0 or (upscale_factor & (upscale_factor - 1)) != 0:
            raise ValueError('`upscale_factor` must be a power of 2.')
        self.upscale_factor = upscale_factor

        self.in_conv = nn.Conv2d(in_channels, num_feats, kernel_size=3, padding=1)
        self.blocks = nn.Sequential(*[ResidualDenseBlock(num_feats) for _ in range(num_layers)])
        
        up_blocks = []
        for _ in range(int(round(math.log2(upscale_factor)))):
            up_blocks.append(nn.Conv2d(num_feats, 4 * num_feats, kernel_size=3, padding=1))
            up_blocks.append(nn.PixelShuffle(2))
        self.up = nn.Sequential(*up_blocks)

        self.out_conv = nn.Conv2d(num_feats, out_channels, kernel_size=1)

    def forward(self, 
                x: torch.Tensor) -> torch.Tensor:
        """Apply 2D spatial upsampling.
        
        Args:
            x (torch.Tensor): shape (B, in_channels, H, W), input feature maps, in_channels.
        
        Returns:
            x (torch.Tensor): shape (B, out_channels, H*upscale_factor, W*upscale_factor), output feature maps.
        """
        shallow_feats = self.in_conv(x)
        x = shallow_feats + self.blocks(shallow_feats)
        x = self.up(x)
        x = self.out_conv(x)
        return x
    
