import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Dict, Any
from types import SimpleNamespace
from .feature_net import FeatureNet
from .depth_net import DepthNet
from .nerf import NeRF
from .decoder_rdn import Decoder
from .bundle_sampler import BundleSampler
from . import utils


class Network(nn.Module):
    def __init__(self, 
                 config: SimpleNamespace):
        super(Network, self).__init__()

        # FPN
        base_channels = config.fpn.base_channels  # base channels of the FeatureNet
        feat_dims = config.fpn.feat_dims  # pyramid feature dimension at each level
        self.feature_net = FeatureNet(base_channels=base_channels, out_channels=feat_dims)

        # MVS
        self.voxel_dim = config.mvs.voxel_dim  # dimension of the voxel feature
        self.depth_net = DepthNet(config)

        # Sampler
        global_num_depth = config.nerf.global_num_depth  # number of global samples per ray
        max_mipmap_level = config.nerf.max_mipmap_level  # maximum mipmap level to use for interpolation
        self.max_num_samples = config.nerf.max_num_samples  # maximum number of samples per ray at each stage
        self.b_size = config.nerf.bundle_size  # ray bundle size (upsampling factor of the rendered feature map)
        if self.b_size <= 0 or (self.b_size & (self.b_size - 1)) != 0:
            raise ValueError('`Bundle size` must be a power of 2.')
        self.inv_depth = config.mvs.inv_depth[-1]  # whether to inverse depth for sampling
        self.is_adaptive = config.nerf.is_adaptive  # whether to use adaptive sampling
        self.sampler = BundleSampler(global_num_depth, max_mipmap_level)

        # NeRF
        self.feat_level = 0  # use the feature map at this level, which is closest to the bundle feature map
        while self.feat_level < len(config.fpn.feat_scales) and config.fpn.feat_scales[self.feat_level] < 1. / self.b_size:
            self.feat_level += 1
        feat_dim = feat_dims[self.feat_level]  # dimension of the used 2D feature
        self.nerf_hidden_dims = config.nerf.nerf_hidden_dims  # hidden dimensions of NeRF
        self.viewdir_agg = config.nerf.viewdir_agg  # whether to use view direction aggregation
        self.render_scale = 1.  # scale of the rendered image, default to 1
        self.nerf = NeRF(self.nerf_hidden_dims, feat_dim, self.voxel_dim, self.viewdir_agg)

        # Decoder
        self.dec_layers = config.nerf.dec_layers  # number of layers of the decoder at each stage
        self.upsampler = Decoder(feat_dim+3+self.voxel_dim, 3, num_feats=64, num_layers=self.dec_layers, upscale_factor=self.b_size)
        self.reweighting = config.nerf.reweighting  # whether to reweight
    
    def render_bundles(self, 
                       rgbs_feat_rgb_dir: torch.Tensor, 
                       vox_feat: torch.Tensor, 
                       z_vals: torch.Tensor, 
                       indices: torch.Tensor, 
                       samples_per_bundle: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Render bundles.

        Args:
            rgbs_feat_rgb_dir: (torch.Tensor) shape (num_views, total_num_samples, 3*b_size*b_size+feat_dim+3+4), rgbs (3*b_size*b_size) 
                + image feature (feat_dim) + rgb (3) + dir (4).
            vox_feat: (torch.Tensor) shape (total_num_samples, voxel_dim), voxel-aligned feature.
            z_vals: (torch.Tensor) shape (total_num_samples,), depth value of sphere centers.
            indices: (torch.Tensor) shape (total_num_samples,), indices of the sampled bundles.
            samples_per_bundle: (torch.Tensor) shape (num_bundles,), number of samples per bundle.
            idx: (int) stage index.
            
        Returns:
            feat: (torch.Tensor) shape (num_bundles, 3*b_size**2+feat_dim+3+voxel_dim), estimated features of bundles.
            depth: (torch.Tensor) shape (num_bundles,), estimated depth map of bundles.
            opacity: (torch.Tensor) shape (num_bundles,), estimated opacity map of bundles.
        """
        # Query density values and features
        sigma, feat = self.nerf(vox_feat, rgbs_feat_rgb_dir)  # (total_num_samples,), (total_num_samples, 3*b_size**2+feat_dim+3+voxel_dim)

        # Differentiable Volumetric Rendering
        num_bundles = samples_per_bundle.shape[0]
        weights, inverse_indices = utils.render_weight_from_density(sigma, indices, num_bundles)  # (total_num_samples,)
        
        if self.inv_depth:  # to disparity
            z_vals = 1. / z_vals

        feat, depth, opacity = utils.accumulate_value_along_rays(feat, z_vals, weights, indices, num_bundles, inverse_indices)  # (num_bundles, :)
        
        if self.inv_depth:  # to depth
            depth = 1. / depth

        return feat, depth, opacity

    def forward(self,
                batch: Dict[str, Any]) -> Tuple[Dict[str, torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
        """
        Args:
            batch: {'src_views': {'rgb': (torch.Tensor) shape (B, num_views, 3, H_orig, W_orig), source images, 
                                  'extrinsics': (torch.Tensor) shape (B, num_views, 4, 4), source view extrinsics, 
                                  'intrinsics': (torch.Tensor) shape (B, num_views, 3, 3), source view intrinsics}. 
                    'tar_views': {'extrinsics': (torch.Tensor) shape (B, 4, 4), target view extrinsics, 
                                  'intrinsics': (torch.Tensor) shape (B, 3, 3), target view intrinsics}. 
                    'near_far': (torch.Tensor) shape (B, 2), near and far depth values of the target view}. 
                    'render_scale': (float) scale of the rendered image, default to 1. 
            
        Returns:
            ret: {'rgb': (torch.Tensor) shape (B, 3, H_orig, W_orig), rendered RGB image, 
                  'nerf_depth': (torch.Tensor) shape (B, H_orig, W_orig), estimated depth map from NeRF, 
                  'mvs_depth': (torch.Tensor) shape (B, H_orig, W_orig), estimated depth map from MVS,
                  'opacity': (torch.Tensor) shape (B, H_orig, W_orig), estimated opacity map from NeRF}.
            mvs_depths: (List[torch.Tensor]) estimated depth maps from MVS at each stage.
            blend_rgbs: (List[torch.Tensor]) estimated RGB maps at each stage, to supervise the depth estimation.
        """
        # Extract data
        src_views = batch['src_views']
        tar_views = batch['tar_views']
        near_far = batch['near_far']  # (B, 2)
        src_images = src_views['rgb']  # (B, num_views, 3, H_orig, W_orig)
        B, num_views, _, H_orig, W_orig = src_images.shape
        src_exts = src_views['extrinsics']  # (B, num_views, 4, 4)
        src_ints = src_views['intrinsics'].clone()  # (B, num_views, 3, 3)
        tar_exts = tar_views['extrinsics']  # (B, 4, 4)
        tar_ints = tar_views['intrinsics'].clone()  # (B, 3, 3)

        # Rescale if required
        if 'render_scale' in batch:
            self.render_scale = batch['render_scale'][0].item()
        if self.render_scale != 1.:
            src_images = F.interpolate(src_images.flatten(0, 1), scale_factor=self.render_scale, mode='bilinear', align_corners=False).unflatten(dim=0, sizes=(B, num_views))
            H_orig, W_orig = src_images.shape[-2:]
            src_ints[..., :2, :] *= self.render_scale
            tar_ints[:, :2, :] *= self.render_scale
        
        # Extract pyramid features
        ms_feats = self.feature_net(src_images.flatten(0, 1))  # [num_levels * (B * num_views, feat_dim, H, W)]
        ms_feats = [f.unflatten(dim=0, sizes=(B, num_views)) for f in ms_feats]  # [num_levels * (B, num_views, feat_dim, H, W)]

        # Estimate depth
        mvs_depths, depth_range_list, vol_range_list, feat_volume_list, blend_rgbs = \
            self.depth_net(src_images, ms_feats, src_exts, src_ints, tar_exts, tar_ints, near_far)
        depth_range = depth_range_list[-1]  # (B, 2, H_orig * feat_scale, W_orig * feat_scale), near and far depth of the confidence interval
        vol_range = vol_range_list[-1]  # (B, 2, H_orig * feat_scale, W_orig * feat_scale), near and far depth of the cost volume
        feat_volume = feat_volume_list[-1]  # (B, feat_dim, num_depth, H_orig * feat_scale, W_orig * feat_scale), cost volume
        mvs_depth = mvs_depths[-1]  # (B, H_orig * feat_scale, W_orig * feat_scale), estimated depth map from MVS

        # Build rays
        self.sampler.build_rays(tar_exts, tar_ints, (H_orig, W_orig), near_far[:, 0], near_far[:, 1])

        # Multi-scale volumetric rendering
        H, W = H_orig // self.b_size, W_orig // self.b_size  # spatial size of the bundle map
        if depth_range.shape[2:] != (H, W):
            depth_range = F.interpolate(depth_range, size=(H, W), mode='bilinear', align_corners=False)
            vol_range = F.interpolate(vol_range, size=(H, W), mode='bilinear', align_corners=False)
            mvs_depth = F.interpolate(mvs_depth.unsqueeze(1), size=(H, W), mode='nearest').squeeze(1)
        
        # Adaptive bundle sampling
        rays_xyz, uvd, z_vals, ball_radii, indices, samples_per_batch, samples_per_bundle = \
            self.sampler.sample(depth_range, vol_range, self.b_size, self.max_num_samples, self.inv_depth, self.is_adaptive)
        
        # Sphere-based encoding
        img_feat_rgb = ms_feats[self.feat_level]
        if img_feat_rgb.shape[-2:] != (H, W):
            img_feat_rgb = F.interpolate(img_feat_rgb.flatten(0, 1), size=(H, W), mode='bilinear', align_corners=False).unflatten(dim=0, sizes=(B, num_views))  # (B*num_views, feat_dim, H, W)
        img_feat_rgb = torch.cat((img_feat_rgb, 
                                  F.interpolate(src_images.flatten(0, 1), size=(H, W), mode='bilinear', align_corners=False).unflatten(dim=0, sizes=(B, num_views))), 
                                 dim=2)  # (B, num_views, feat_dim+3, H, W)
        rgbs_feat_rgb_dir, vox_feat = \
            self.sampler.encode(src_images, img_feat_rgb, feat_volume, rays_xyz, uvd, ball_radii, src_exts, src_ints, tar_exts, samples_per_batch)
        
        # Render bundles
        bundle_feat, bundle_depth, bundle_opacity = self.render_bundles(rgbs_feat_rgb_dir, vox_feat, z_vals, indices, samples_per_bundle)
        nerf_feat = bundle_feat.view(B, H, W, -1).permute(0, 3, 1, 2)  # (B, 3*b_size**2+feat_dim1+3+voxel_dim, H, W)
        nerf_depth = bundle_depth.view(B, H, W)
        nerf_opacity = bundle_opacity.view(B, H, W)
        num_rays_per_bundle = 3 * self.b_size ** 2
        rgb_c = self.upsampler(nerf_feat[:, num_rays_per_bundle:])  # (B, 3, H_orig, W_orig)
        rgb_f = F.pixel_shuffle(nerf_feat[:, :num_rays_per_bundle], self.b_size)  # (B, 3, H_orig, W_orig)
        nerf_depth = F.interpolate(nerf_depth.unsqueeze(1), scale_factor=self.b_size, mode='bilinear', align_corners=False).squeeze(1)  # (B, H_orig, W_orig)
        nerf_opacity = F.interpolate(nerf_opacity.unsqueeze(1), scale_factor=self.b_size, mode='bilinear', align_corners=False).squeeze(1)  # (B, H_orig, W_orig)
        
        # Merge coarse and fine RGB features, and reweight if required (for cross dataset view synthesis)
        img = rgb_c + rgb_f  # (B, 3, H_orig, W_orig)
        if self.reweighting:
            img = 0.5 * (img + rgb_f)
            
        ret = {'rgb': img, 
               'nerf_depth': nerf_depth, 
               'mvs_depth': mvs_depth, 
               'opacity': nerf_opacity}

        return ret, mvs_depths, blend_rgbs