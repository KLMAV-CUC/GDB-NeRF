import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple


class ResidualDenseBlock(nn.Module):
    def __init__(self, 
                 num_feats: int, 
                 growth_rate: int = 32) -> None:
        super(ResidualDenseBlock, self).__init__()
        self.conv1 = nn.Conv2d(num_feats, growth_rate, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(num_feats + growth_rate, growth_rate, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(num_feats + 2 * growth_rate, num_feats, kernel_size=3, padding=1)

    def forward(self, 
                x: torch.Tensor) -> torch.Tensor:
        x1 = F.relu(self.conv1(x), inplace=True)
        x2 = F.relu(self.conv2(torch.cat([x, x1], dim=1)), inplace=True)
        x3 = self.conv3(torch.cat([x, x1, x2], dim=1))
        return x + x3


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


class BundleDecoder(nn.Module):
    def __init__(self, 
                 feat_scales: List[float],
                 feat_dims: List[int],) -> None:
        """Bundle decoder for NeRF."""
        super(BundleDecoder, self).__init__()
        
        self.feat_dims = feat_dims
        self.num_levels = len(feat_dims)
        self.upscales = [int(feat_scales[i+1] / feat_scales[i]) if i < self.num_levels-1 else int(1. / feat_scales[i]) for i in range(self.num_levels)]

        # Modules
        self.decoders = nn.ModuleList()
        for level in range(self.num_levels-1):
            self.decoders.append(
                Decoder(
                    in_channels=self.feat_dims[level]+3, 
                    out_channels=self.feat_dims[level+1]+3, 
                    num_feats=64, 
                    num_layers=2, 
                    upscale_factor=self.upscales[level])
            )
        self.decoders.append(
            Decoder(
                in_channels=self.feat_dims[-1]+3, 
                out_channels=3, 
                num_feats=64, 
                num_layers=2, 
                upscale_factor=self.upscales[-1])
        )
    
    def forward(self, 
                bundle_feat: List[torch.Tensor], 
                bundle_depth: List[torch.Tensor], 
                bundle_opacity: List[torch.Tensor], 
                mask: List[torch.Tensor], 
                reweighting: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Decode bundle features and render images.

        Args:   
            bundle_feat: (List[torch.Tensor]) shape [num_levels * (num_bundles, 3*b_size**2+feat_dim+3)], estimated features of bundles.
            bundle_depth: (List[torch.Tensor]) shape [num_levels * (num_bundles,)], estimated depth map of bundles.
            bundle_opacity: (List[torch.Tensor]) shape [num_levels * (num_bundles,)], estimated opacity map of bundles.
            mask: (List[torch.Tensor]) shape [num_levels * (B, H, W)], mask of valid samples.
            
        Returns:
            img: (torch.Tensor) shape (B, 3, H_orig, W_orig), rendered image.
            depth: (torch.Tensor) shape (B, H_orig, W_orig), rendered depth map.
            opacity: (torch.Tensor) shape (B, H_orig, W_orig), rendered opacity map.
        """
        device = bundle_feat[0].device
        dtype = bundle_feat[0].dtype
        B = bundle_feat[0].shape[0]  # batch size

        if self.training:  # soft mask
            for level in range(self.num_levels):
                H, W = mask[level].shape[-2:]
                bundle_feat[level] = bundle_feat[level].view(B, H, W, -1).permute(0, 3, 1, 2)
                bundle_depth[level] = bundle_depth[level].view(B, 1, H, W)
                bundle_opacity[level] = bundle_opacity[level].view(B, 1, H, W)
        else:  # binary mask
            for level in range(self.num_levels):
                mask_level = mask[level]
                H, W = mask_level.shape[-2:]
                feat_level = torch.zeros(B, H, W, bundle_feat[level].shape[-1], dtype=dtype, device=device)
                feat_level[mask_level] = bundle_feat[level]
                bundle_feat[level] = feat_level.permute(0, 3, 1, 2)
                depth_level = torch.zeros(B, H, W, dtype=dtype, device=device)
                depth_level[mask_level] = bundle_depth[level]
                bundle_depth[level] = depth_level.unsqueeze(1)
                opacity_level = torch.zeros(B, H, W, dtype=dtype, device=device)
                opacity_level[mask_level] = bundle_opacity[level]
                bundle_opacity[level] = opacity_level.unsqueeze(1)
        
        # Coarest level
        ch = self.feat_dims[0] + 3
        rgb = bundle_feat[0][:, :-ch]  # (B, 3*b_size[0]**2, H[0], W[0])
        rgb = F.pixel_shuffle(rgb, self.upscales[0])  # (B, 3*b_size[1]**2, H[1], W[1])
        feat = bundle_feat[0][:, -ch:]  # (B, feat_dim[0]+3, H[0], W[0])
        feat = self.decoders[0](feat)  # (B, feat_dim[1]+3, H[1], W[1])
        
        depth = F.interpolate(bundle_depth[0], scale_factor=self.upscales[0], mode='nearest')  # (B, 1, H[1], W[1])
        opacity = F.interpolate(bundle_opacity[0], scale_factor=self.upscales[0], mode='nearest')  # (B, 1, H[1], W[1])

        for level in range(1, self.num_levels):
            ch = self.feat_dims[level] + 3
            rgb_level = bundle_feat[level][:, :-ch]  # (B, 3*b_size[level]**2, H[level], W[level])
            feat_level = bundle_feat[level][:, -ch:]  # (B, feat_dim[level]+3, H[level], W[level])
            mask_level = mask[level].unsqueeze(1).to(dtype)  # (B, 1, H[level], W[level])
            rgb = rgb * (1 - mask_level) + rgb_level * mask_level
            rgb = F.pixel_shuffle(rgb, self.upscales[level])  # (B, 3*b_size[level+1]**2, H[level+1], W[level+1]) or (B, 3, H_orig, W_orig)
            feat = feat * (1 - mask_level) + feat_level * mask_level
            feat = self.decoders[level](feat)  # (B, feat_dim[level+1]+3, H[level+1], W[level+1]) or (B, 3, H_orig, W_orig)

            depth = depth * (1 - mask_level) + bundle_depth[level] * mask_level
            depth = F.interpolate(depth, scale_factor=self.upscales[level], mode='nearest')  # (B, 1, H[level+1], W[level+1]) or (B, 3, H_orig, W_orig)
            opacity = opacity * (1 - mask_level) + bundle_opacity[level] * mask_level
            opacity = F.interpolate(opacity, scale_factor=self.upscales[level], mode='nearest')  # (B, 1, H[level+1], W[level+1]) or (B, 3, H_orig, W_orig)
        
        img = feat + rgb  # (B, 3, H_orig, W_orig)
        if reweighting:
            img = 0.5 * (img + rgb)

        return img, depth.squeeze(1), opacity.squeeze(1)