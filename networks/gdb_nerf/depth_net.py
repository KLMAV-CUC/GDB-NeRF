import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List
from types import SimpleNamespace
from operator import itemgetter
from .cost_reg_net import CostRegNet, CostRegNet_small


class DepthNet(nn.Module):
    def __init__(self, 
                 config: SimpleNamespace) -> None:
        super(DepthNet, self).__init__()
        """CasMVSNet for depth estimation."""

        # Hyperparameters
        base_channels = config.fpn.base_channels  # base channels of CostRegNet
        self.vol_levels = config.mvs.vol_levels  # which level of the feature maps to use for constructing the cost volume at each stage
        self.vol_scales = config.mvs.vol_scales  # scale of the cost volume at each stage
        self.num_stages = len(self.vol_levels)  # number of stages
        feat_scales = itemgetter(*self.vol_levels)(config.fpn.feat_scales)  # feature scales used at each stage
        self.feat_scales = feat_scales if isinstance(feat_scales, (list, tuple)) else (feat_scales,)
        feat_dims = itemgetter(*self.vol_levels)(config.fpn.feat_dims)  # feature dimensions used at each stage
        self.feat_dims = feat_dims if isinstance(feat_dims, (list, tuple)) else (feat_dims,)
        self.ci_scales = config.mvs.ci_scales  # scale of confidence interval at each stage
        voxel_dim = config.mvs.voxel_dim  # dimension of the voxel feature
        self.num_depth = config.mvs.num_depth  # number of depth hypothesis at each stage
        self.inv_depth = config.mvs.inv_depth  # whether to inverse depth (disparity) at each stage

        # Modules
        self.cost_regs = nn.ModuleList(
            [CostRegNet_small(self.feat_dims[self.vol_levels[0]], voxel_dim, base_channels)]
        )
        for i in range(1, self.num_stages):
            self.cost_regs.append(
                CostRegNet(self.feat_dims[self.vol_levels[i]], voxel_dim, base_channels)
            )

        # NeRFs only for training
        if self.training:
            self.num_samples = config.mvs.num_samples  # number of samples along rays of NeRFs for training
            self.nerf_hidden_dims = config.nerf.nerf_hidden_dims  # hidden dimensions of NeRF
            self.chunk_size = config.nerf.chunk_size  # chunk size of NeRFs
            self.viewdir_agg = config.nerf.viewdir_agg  # whether to use view direction aggregation
            self.nerfs = nn.ModuleList()
            for i in range(self.num_stages - 1):
                self.nerfs.append(NeRF(self.nerf_hidden_dims, voxel_dim, self.feat_dims[i], self.viewdir_agg))
    
    def _render_rays(self, 
                    rays: torch.Tensor, 
                    feat_volume: torch.Tensor, 
                    img_feat_rgb: torch.Tensor, 
                    src_exts: torch.Tensor, 
                    src_ints: torch.Tensor, 
                    tar_exts: torch.Tensor, 
                    idx: int) -> torch.Tensor:
        """Render rays.

        Args:
            rays: (torch.Tensor) shape (B, num_rays, 14), rays origin (3), unnormalized direction (3), normalized pixel coordinates (2), 
                ray depth range (2), volume depth range (2), target pixel radius (1) and cos of ray angle with z-axis (1).
            near_far: (torch.Tensor) shape (B, 2), near and far depth range.
            feat_volume: (torch.Tensor) shape (B, feat_dim, num_depth, H, W), cost volume.
            img_feat_rgb: (torch.Tensor) shape (B, num_views, feat_dim+3, H, W), source feature (feat_dim) + rgb (3).
            src_exts: (torch.Tensor) shape (B, num_views, 4, 4), source extrinsic matrices (w2c).
            src_ints: (torch.Tensor) shape (B, num_views, 3, 3), source intrinsic matrices.
            tar_exts: (torch.Tensor) shape (B, 4, 4), target extrinsic matrix (w2c).
            idx: (int) index of the current stage.
        
        Returns:
            rgb_map: (torch.Tensor) shape (B, num_rays, 3), estimated RGB color of a ray.
        """
        device = rays.device
        B, num_rays = rays.shape[:2]

        # Sample along the ray from near to far at equal depth intervals
        rays_o, rays_d, uv = rays[..., :3], rays[..., 3:6], rays[..., 6:8]
        ray_near, ray_far, vol_near, vol_far = rays[..., 8:9], rays[..., 9:10], rays[..., 10:11], rays[..., 11:12]
        if self.inv_depth[idx]:  # to disparity
            ray_near, ray_far = 1. / ray_far, 1. / ray_near
            vol_near, vol_far = 1. / vol_far, 1. / vol_near
        t_vals = ray_near + (ray_far - ray_near) * torch.linspace(0., 1., self.num_samples[idx]+1, device=device, dtype=torch.float32)
        t_starts = t_vals[..., :-1]  # (B, num_rays, num_samples)
        t_ends = t_vals[..., 1:]  # (B, num_rays, num_samples)
        z_vals = 0.5 * (t_starts + t_ends)  # (B, num_rays, num_samples)
        d = 2 * (z_vals - vol_near) / (vol_far - vol_near) - 1.  # z-axis value, scaled to [-1, 1]
        uvd = torch.cat([uv[..., None, :].expand(-1, -1, self.num_samples[idx], -1), d[..., None]], dim=-1)  # (B, num_rays, num_samples, 3)
        if self.inv_depth[idx]:  # to depth
            z_vals = 1. / z_vals
        world_xyz = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., None]  # (B, num_rays, num_samples, 3)

        # Extract voxel and image features
        vox_feat = F.grid_sample(feat_volume, uvd.view(B, -1, 1, 1, 3), mode='bilinear', padding_mode='border', align_corners=False)  # (B, voxel_dim, num_rays * num_samples, 1, 1)
        vox_feat = vox_feat.flatten(2, -1).permute(0, 2, 1)  # (B, num_rays * num_samples, voxel_dim)
        img_feat_rgb_dir = get_img_feat_vectorized(img_feat_rgb, world_xyz, src_exts, src_ints, tar_exts)  # (B, num_rays * num_samples, num_views, feat_dim+3+4)

        # Query NeRFs for each sample
        sigma = torch.zeros(B, num_rays * self.num_samples[idx], device=device)  # (B, num_rays * num_samples)
        rgb = torch.zeros(B, num_rays * self.num_samples[idx], 3, device=device)  # (B, num_rays * num_samples, 3)
        chunk_size = self.chunk_size * self.num_samples[idx]
        for i in range(0, num_rays, chunk_size):
            sub_sigma, sub_rgb = self.nerfs[idx](vox_feat[:, i:i + chunk_size], img_feat_rgb_dir[:, i:i + chunk_size])  # (B, chunk_size), (B, chunk_size, 3)
            sigma[:, i:i + chunk_size] = sub_sigma
            rgb[:, i:i + chunk_size] = sub_rgb
        sigma = sigma.unflatten(dim=1, sizes=(num_rays, self.num_samples[idx]))
        rgb = rgb.unflatten(dim=1, sizes=(num_rays, self.num_samples[idx]))
        
        # Volume rendering
        alpha = 1. - torch.exp(-sigma)  # (B, num_rays, num_samples)
        T = torch.cumprod(1. - alpha + 1e-10, dim=-1)[..., :-1]  # (B, num_rays, num_samples-1)
        T = torch.cat([torch.ones_like(alpha[..., 0:1]), T], dim=-1)  # (B, num_rays, num_samples)
        weights = alpha * T  # (B, num_rays, num_samples)
        # weights = F.normalize(weights, p=1, dim=-1)  # (B, num_rays, num_samples)
        rgb_map = torch.sum(weights[..., None] * rgb, dim=-2)  # (B, num_rays, 3)

        return rgb_map

    def forward(self,
                src_images: torch.Tensor, 
                ms_feats: List[torch.Tensor], 
                src_exts: torch.Tensor, 
                src_ints: torch.Tensor, 
                tar_exts: torch.Tensor, 
                tar_ints: torch.Tensor, 
                near_far: torch.Tensor) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
        """Render the target view with given camera parameters.
        
        Args:
            src_images: (torch.Tensor) shape (B, num_views, 3, H_ori, W_ori), multiscale source features.
            ms_feats: (List[torch.Tensor]) shape [num_stages * (B, num_views, feat_dim, Hi, Wi)], source feature maps at each stage.
            src_exts: (torch.Tensor) shape (B, num_views, 4, 4), source extrinsic matrices (w2c).
            src_ints: (torch.Tensor) shape (B, num_views, 3, 3), camera intrinsics at each level.
            tar_exts: (torch.Tensor) shape (B, 4, 4), target extrinsic matrix (w2c).
            tar_ints: (torch.Tensor) shape (B, 3, 3), target camera intrinsics at each level.
            near_far: (torch.Tensor) shape (B, 2, 1, 1), depth range of the target view.

        Returns:
            depth_predictions: (List[torch.Tensor]) shape [num_stages * (B, Hi, Wi)], depth predictions at each stage.
            depth_range_list: (List[torch.Tensor]) shape [num_stages * (B, 2, Hi, Wi)], depth range of the confidence interval at each stage.
            vol_range_list: (List[torch.Tensor]) shape [num_stages * (B, num_depth, Hi, Wi)], depth range of the cost volume at each stage.
            feat_volume_list: (List[torch.Tensor]) shape [num_stages * (B, voxel_dim, num_depth, Hi, Wi)], feature volume at each stage.
            rgb_predictions: (List[torch.Tensor]) shape [(num_stages - 1) * (B, 3, Hi, Wi)], RGB predictions at each stage for training.
        """
        # Extract features
        B, num_views, _, H_ori, W_ori = src_images.shape

        feat_volume_list = []
        depth_predictions = []
        depth_range_list = []
        vol_range_list = []
        rgb_predictions = []  # for training

        depth_range = near_far[..., None, None]  # (B, 2, 1, 1)

        for idx in range(self.num_stages):
            feats_stage = ms_feats[self.vol_levels[idx]]  # feature maps at current stage, (B, num_views, feat_dim, H_ori * feat_scales, W_ori * feat_scales)
            
            # Rescale intrinsics
            src_ints_stage = src_ints.clone()
            src_ints_stage[..., :2, :] *= self.feat_scales[idx]
            tar_ints_stage = tar_ints.clone()
            tar_ints_stage[:, :2, :] *= self.vol_scales[idx]
            Hi, Wi = int(H_ori * self.vol_scales[idx]), int(W_ori * self.vol_scales[idx])  # target size of current stage

            # Build cost volume with equal depth/disparity (inverse depth) spacing
            depth_values = get_depth_values(depth_range, self.num_depth[idx], self.inv_depth[idx])  # (B, num_depth, 1, 1) or (B, num_depth, Hi, Wi)
            depth_values = depth_values.expand(-1, -1, Hi, Wi)
            feat_volume = build_feature_volume(feats_stage, src_exts, src_ints_stage, tar_exts, tar_ints_stage, depth_values, self.inv_depth[idx])  # (B, feat_dim, num_depth, Hi, Wi)

            # Cost volume regularization and depth regression
            feat_volume, depth_prob = self.cost_regs[idx](feat_volume)
            depth, ci = depth_regression(depth_values, depth_prob, self.ci_scales[idx], self.inv_depth[idx])

            # Update depth range for next stage
            depth_range = ci
            
            depth_predictions.append(depth.squeeze(1))
            depth_range_list.append(depth_range)
            vol_range_list.append(depth_values[:, [0, -1]])
            feat_volume_list.append(feat_volume)

            if idx < self.num_stages - 1:
                # Render target view with NeRF for training
                if self.training:
                    rays = build_rays(tar_exts, tar_ints_stage, depth_range, vol_range_list[-1])  # (B, Hi * Wi, 15)
                    img_feat_rgb = torch.cat(
                        (feats_stage, 
                         F.interpolate(src_images.flatten(0, 1), scale_factor=self.feat_scales[idx], mode='bilinear', align_corners=False).unflatten(dim=0, sizes=(B, num_views))), 
                        dim=2
                    )  # (B, num_views, feat_dim+3, H_ori * feat_scales, W_ori * feat_scales)
                    ray_rgbs = self._render_rays(rays, feat_volume, img_feat_rgb, src_exts, src_ints_stage, tar_exts, idx)
                    rgb_predictions.append(ray_rgbs.permute(0, 2, 1).view(B, -1, Hi, Wi))
                
                # Upsample depth range for next stage
                up_scale = self.vol_scales[idx + 1] / self.vol_scales[idx]
                depth_range = F.interpolate(depth_range, scale_factor=up_scale, mode='bilinear', align_corners=False)
                 
        return depth_predictions, depth_range_list, vol_range_list, feat_volume_list, rgb_predictions


class NeRF(nn.Module):
    def __init__(self, 
                 hid_dim: int = 64, 
                 voxel_dim: int = 8, 
                 feat_dim: int = 16, 
                 viewdir_agg: bool = True) -> None:
        """Generalizable NeRF (MVSNeRF) based on agregated 2D, 3D features and directions.
        """
        super(NeRF, self).__init__()
        self.hid_dim = hid_dim
        self.viewdir_agg = viewdir_agg
        
        # View direction aggregation
        if viewdir_agg:
            self.view_fc = nn.Sequential(
                nn.Linear(4, feat_dim+3), 
                nn.ReLU(inplace=True)
            )
        self.global_fc = nn.Sequential(
            nn.Linear((feat_dim+3)*3, 32), 
            nn.ReLU(inplace=True)
        )
        self.agg_w_fc = nn.Sequential(
            nn.Linear(32, 1), 
            nn.ReLU(inplace=True)
        )
        self.fc = nn.Sequential(
            nn.Linear(32, 16), 
            nn.ReLU(inplace=True)
        )

        # Color aggregation
        self.lr0 = nn.Sequential(
            nn.Linear(voxel_dim+16, hid_dim), 
            nn.ReLU(inplace=True)
        )
        self.sigma = nn.Sequential(
            nn.Linear(hid_dim, 1), 
            nn.Softplus()
        )
        self.color = nn.Sequential(
            nn.Linear(hid_dim+voxel_dim+16+feat_dim+3+4, hid_dim), 
            nn.ReLU(inplace=True), 
            nn.Linear(hid_dim, 1), 
            nn.ReLU(inplace=True)
        )
    
    def agg_viewdir(self, 
                    img_feat_rgb_dir: torch.Tensor) -> torch.Tensor:
        """Aggregate view direction features.

        Args:
            img_feat_rgb_dir: (torch.Tensor) shape (B, num_rays * num_points, num_views, feat_dim + rgbs (3) + dir (4)).
        Returns:
            _: (torch.Tensor) shape (B, num_rays * num_points, 16)
        """
        num_views = img_feat_rgb_dir.shape[-2]
        img_feat_rgb = img_feat_rgb_dir[..., :-4]  # (B, num_rays * num_points, num_views, feat_dim+3)
        if self.viewdir_agg:
            view_feat = self.view_fc(img_feat_rgb_dir[..., -4:])  # (B, num_rays * num_points, num_views, feat_dim+3)
            img_feat_rgb = img_feat_rgb + view_feat

        var_feat, avg_feat = torch.var_mean(img_feat_rgb, dim=-2, keepdim=True)
        var_feat = var_feat.expand(-1, -1, num_views, -1)
        avg_feat = avg_feat.expand(-1, -1, num_views, -1)

        feat = torch.cat([img_feat_rgb, var_feat, avg_feat], dim=-1)  # (B, num_rays * num_points, num_views, (feat_dim+3)*3)
        global_feat = self.global_fc(feat)  # (B, num_rays * num_points, num_views, 32)
        agg_w = F.softmax(self.agg_w_fc(global_feat), dim=-2)  # (B, num_rays * num_points, num_views, 1)
        im_feat = (global_feat * agg_w).sum(dim=-2)  # (B, num_rays * num_points, 32)
        
        return self.fc(im_feat)

    def forward(self, 
                vox_feat: torch.Tensor, 
                img_feat_rgb_dir: torch.Tensor) -> torch.Tensor:
        """Predict color/feat and sigma for each ray sample.

        Args:
            vox_feat: (torch.Tensor) shape (B, num_rays * num_samples, vox_dim), voxel features from cost volume.
            img_feat_rgb_dir: (torch.Tensor) shape (B, num_rays * num_samples, num_views, feat_dim + rgbs (3) + dir (4)).
        Returns:
            sigma: (torch.Tensor) shape (B, num_rays * num_samples).
            rgb: (torch.Tensor) shape (B, num_rays * num_samples, rgbs (3)).
        """
        num_views = img_feat_rgb_dir.shape[-2]
        img_feat = self.agg_viewdir(img_feat_rgb_dir)  # (B, num_rays * num_samples, 16)
        vox_img_feat = torch.cat((vox_feat, img_feat), dim=-1)  # (B, num_rays * num_samples, vox_dim+16)
        x = self.lr0(vox_img_feat)  # (B, num_rays * num_samples, hid_dim)
        sigma = self.sigma(x)  # (B, num_rays * num_samples, 1)

        x = torch.cat((x, vox_img_feat), dim=-1)  # (B, num_rays * num_samples, hid_dim+vox_dim+16)
        x = torch.stack(num_views * [x], dim=-2)  # (B, num_rays * num_samples, num_views, hid_dim+vox_dim+16)
        x = torch.cat((x, img_feat_rgb_dir), dim=-1)  # (B, num_rays * num_samples, num_views, hid_dim+vox_dim+16+feat_dim+3+4)
        weight = F.softmax(self.color(x), dim=-2)  # (B, num_rays * num_samples, num_views, 1)
        rgb = torch.sum((img_feat_rgb_dir[..., -7:-4] * weight), dim=-2)  # (B, num_rays * num_samples, 3) 

        return sigma.squeeze(-1), rgb


def build_rays(tar_exts: torch.Tensor, 
               tar_ints: torch.Tensor, 
               ray_range: torch.Tensor, 
               vol_range: torch.Tensor) -> torch.Tensor:
    """Builds rays for NeRF.

	Args:
        tar_exts: (torch.Tensor) shape (B, 4, 4), target camera extrinsics (w2c).
        tar_ints: (torch.Tensor) shape (B, 3, 3), target camera intrinsics.
        ray_range: (torch.Tensor) shape (B, 2, H, W), depth/disparity range of each ray.
        vol_range: (torch.Tensor) shape (B, 2, H, W), depth/disparity range of depth hypothesis.

	Returns:
        rays: (torch.Tensor) shape (B, H * W, 12), rays origin (3), unnormalized direction (3), normalized pixel coordinates (2),
            ray depth range (2), volume depth range (2).
	"""
    device = tar_exts.device
    B, _, H, W = ray_range.shape

    x, y = torch.meshgrid(torch.arange(W, dtype=torch.float32, device=device) + 0.5, 
                          torch.arange(H, dtype=torch.float32, device=device) + 0.5, 
                          indexing='xy')  # (H, W)
        
    x = x.flatten()  # (H * W)
    y = y.flatten()  # (H * W)
    xyz = torch.stack((x, y, torch.ones_like(x)), dim=1)  # (H * W, 3)
        
    # To world coordinates
    c2w = torch.inverse(tar_exts)
    rays_o = c2w[:, None, :3, 3].expand(-1, H * W, -1)  # (B, H * W, 3)
    rays_d = xyz @ (c2w[:, :3, :3] @ torch.inverse(tar_ints)).transpose(-2, -1)  # (B, H * W, 3), unnormalized
        
    uv = torch.stack((2 * x / W - 1, 2 * y / H - 1), dim=-1).unsqueeze(0).expand(B, -1, -1)  # (B, H * W, 2), scaled to [-1, 1]
        
    # Compute depth range
    ray_near_far = ray_range.permute(0, 2, 3, 1).flatten(1, 2)  # (B, H * W, 2)
    vol_near_far = vol_range.permute(0, 2, 3, 1).flatten(1, 2)  # (B, H * W, 2), depth range of the cost volume

    rays = torch.cat((rays_o, rays_d, uv, ray_near_far, vol_near_far), dim=-1)  # (B, H * W, 12)

    return rays


def get_img_feat_vectorized(img_feat_rgb: torch.Tensor, 
                            world_xyz: torch.Tensor, 
                            src_exts: torch.Tensor, 
                            src_ints: torch.Tensor, 
                            tar_exts: torch.Tensor) -> torch.Tensor:
    """Get the corresponding image features from the source image according to the world coordinates.

    Args:
        img_feat_rgb: (torch.Tensor) shape (B, num_views, feat_dim+3, H, W), source feature (feat_dim) + rgb (3).
        world_xyz: (torch.Tensor) shape (B, num_rays, num_samples, 3), 3D world coordinates of samples.
        src_exts: (torch.Tensor) shape (B, num_views, 4, 4), source extrinsic matrices (w2c).
        src_ints: (torch.Tensor) shape (B, num_views, 3, 3), source intrinsic matrices.
        tar_exts: (torch.Tensor) shape (B, 4, 4), target extrinsic matrix (w2c).
    
    Returns:
        img_feat_rgb_dir: (torch.Tensor) shape (B, num_rays * num_samples, num_views, feat_dim+3+4), image feature + rgb (3) + dir (4).
    """
    B, num_views, _, H, W = img_feat_rgb.shape
    num_rays, num_samples = world_xyz.shape[1:3]

    # Compute camera coordinates in source views
    world_xyz = world_xyz.view(B, -1, 3)  # (B, num_rays * num_samples, 3)
    world_xyz = torch.cat([world_xyz, torch.ones_like(world_xyz[..., :1])], dim=-1)  # (B, num_rays * num_samples, 4)
        
    xyz_img = torch.matmul(torch.matmul(world_xyz[:, None], src_exts.transpose(-2, -1))[..., :3], src_ints.transpose(-2, -1))  # (B, num_views, num_rays * num_samples, 3)
    invalid = xyz_img[..., 2] < 1e-8  # (B, num_views, num_rays * num_samples)
    grid = xyz_img[..., :2] / xyz_img[..., 2:3]  # (B, num_views, num_rays * num_samples, 2)
    grid[..., 0], grid[..., 1] = 2 * grid[..., 0] / W - 1, 2 * grid[..., 1] / H - 1  # normalize to [-1, 1]
    grid[invalid] = -99.

    # grid = xyz_img[..., :2] / xyz_img[..., 2:3].clamp_min(1e-8)  # (B, num_views, num_rays * num_samples, 2)
    # grid[..., 0], grid[..., 1] = 2 * grid[..., 0] / W - 1, 2 * grid[..., 1] / H - 1  # normalize to [-1, 1]
    
    # Sample features using grid
    feats = F.grid_sample(img_feat_rgb.flatten(0, 1), grid.view(B * num_views, -1, 1, 2), 
                          mode='bilinear', padding_mode='border', align_corners=False)  # (B * num_views, feat_dim+3, num_rays * num_samples, 1)
    feats = feats.view(B, num_views, -1, num_rays * num_samples).permute(0, 3, 1, 2)  # (B, num_rays * num_samples, num_views, feat_dim+3)
    
    # Compute ray differences
    tar_cam_xyz = torch.inverse(tar_exts)[:, None, :3, 3]  # (B, 1, 3)
    src_cam_xyz = torch.inverse(src_exts)[..., :3, 3]  # (B, num_views, 3)
    tar_diff = F.normalize(world_xyz[..., :3] - tar_cam_xyz, p=2.0, dim=-1)  # (B, num_rays * num_samples, 3)
    src_diff = F.normalize(world_xyz[..., None, :3] - src_cam_xyz[:, None], p=2.0, dim=-1)  # (B, num_rays * num_samples, num_views, 3)
    
    # Compute ray direction and dot product
    ray_diff_direction = F.normalize(tar_diff[..., None, :] - src_diff, p=2.0, dim=-1)  # (B, num_rays * num_samples, num_views, 3)
    ray_diff_dot = torch.sum(tar_diff[..., None, :] * src_diff, dim=-1, keepdim=True)  # (B, num_rays * num_samples, num_views, 1)
    ray_diff = torch.cat([ray_diff_direction, ray_diff_dot], dim=-1)  # (B, num_rays * num_samples, num_views, 4)
    
    # Concatenate features and ray differences
    img_feat_rgb_dir = torch.cat([feats, ray_diff], dim=-1)  # (B, num_rays * num_samples, num_views, feat_dim+3+4)
    
    return img_feat_rgb_dir


def get_depth_values(near_far: torch.Tensor, 
                     num_depth: int, 
                     inv_depth: bool) -> torch.Tensor:
    """Get depth/disparity values for depth hypothesis from near to far.
        
    Args:
        near_far: (torch.Tensor) shape (B, 2, H, W), near and far depth value.
        num_depth: (int) number of depth values.
        inv_depth: (bool) whether to inverse depth (disparity).
            
    Returns:
        depth_values: (torch.Tensor) shape (B, num_depth, H, W), depth\disparity values of depth hypothesis from near to far.
    """
        
    depth_near, depth_far = near_far[:, :1], near_far[:, -1:]
        
    if inv_depth:  # to disparity (inverse depth)
        depth_near = 1. / depth_near
        depth_far = 1. / depth_far
    depth_values = depth_near + (depth_far - depth_near) * \
        torch.linspace(0., 1., num_depth, device=depth_near.device).view(1, num_depth, 1, 1)
        
    return depth_values


def build_feature_volume(src_feat: torch.Tensor, 
                         src_exts: torch.Tensor, 
                         src_ints: torch.Tensor, 
                         tar_exts: torch.Tensor, 
                         tar_ints: torch.Tensor, 
                         depth_values: torch.Tensor, 
                         inv_depth: bool) -> torch.Tensor:
    """Build feature volume.
        
    Args:
        src_feat: (torch.Tensor) shape (B, num_views, C, Hs, Ws), source feature maps.
        src_exts: (torch.Tensor) shape (B, num_views, 4, 4), source extrinsics (w2c).
        src_ints: (torch.Tensor) shape (B, num_views, 3, 3), source intrinsics.
        tar_exts: (torch.Tensor) shape (B, 4, 4), target extrinsic (w2c).
        tar_ints: (torch.Tensor) shape (B, 3, 3), target intrinsic.
        depth_values: (torch.Tensor) shape (B, num_depth, Ht, Wt), depth/disparity values of depth hypothesis from near to far.
        inv_depth: (bool) whether to inverse depth (disparity).
            
    Returns:
        volume_variance: (torch.Tensor) shape (B, C, num_depth, Ht, Wt), feature volume using variance metric.
    """
    dtype=src_feat.dtype
    device=src_feat.device
    B, num_views, _, Hs, Ws = src_feat.shape
    num_depth, Ht, Wt = depth_values.shape[1:]
    if inv_depth:  # to depth
        depth_values = 1. / depth_values
        
    # Get projection matrices
    src_projs = torch.matmul(src_ints, src_exts[..., :3, :])  # (B, num_views, 3, 4)
    tar_proj = torch.matmul(tar_ints, tar_exts[..., :3, :])  # (B, 3, 4)
    tar_proj = F.pad(tar_proj, (0, 0, 0, 1), value=0.0)  # (B, 4, 4)
    tar_proj[..., 3, 3] = 1.0
    proj_mats = torch.matmul(src_projs, torch.inverse(tar_proj).unsqueeze(1)).view(-1, 3, 4)  # (B * num_views, 3, 4)

    # create grid from the ref frame
    x, y = torch.meshgrid(torch.arange(Wt, dtype=dtype, device=device) + 0.5, 
                          torch.arange(Ht, dtype=dtype, device=device) + 0.5, 
                          indexing='xy')
    xyz = torch.stack([x, y, torch.ones_like(x)], dim=0).reshape(1, 3, Ht * Wt)
        
    # Apply differentiable homography warping
    depth_values = depth_values.view(B, 1, num_depth, -1).expand(-1, num_views, -1, -1).reshape(B * num_views, 1, num_depth, -1)
    xyz = torch.matmul(proj_mats[..., :3], xyz).unsqueeze(2) * depth_values + proj_mats[..., 3:, None]  # (B * num_views, 3, num_depth, Ht * Wt)
    xyz = xyz.permute(0, 2, 3, 1).contiguous()  # (B * num_views, num_depth, Ht * Wt, 3)
    grid = xyz[..., :2] / xyz[..., 2:3].clamp_min(1e-6)  # (B * num_views, num_depth, Ht * Wt, 2)
    grid[..., 0], grid[..., 1] = 2 * grid[..., 0] / Ws - 1, 2 * grid[..., 1] / Hs - 1  # scale to [-1, 1]
        
    warped_volume = F.grid_sample(src_feat.flatten(0, 1), grid, mode='bilinear', padding_mode='zeros', align_corners=False)  # (B * num_views, C, num_depth, Ht * Wt)
    warped_volume = warped_volume.view(B, num_views, -1, num_depth, Ht, Wt)
    volume_variance = torch.var(warped_volume, dim=1, unbiased=False)  # (B, C, num_depth, Ht, Wt)
        
    return volume_variance


def depth_regression(depth_values: torch.Tensor, 
                     depth_prob: torch.Tensor, 
                     ci_scale: float, 
                     inv_depth: bool) -> Tuple[torch.Tensor, torch.Tensor]:
    """Perform depth regression on cost volume.
    
    Args:
        depth_values: (torch.Tensor) shape (B, num_depth, H, W), depth/disparity values of depth hypothesis from near to far.
        depth_prob: (torch.Tensor) shape (B, num_depth, H, W), depth probability of each depth hypothesis.
        ci_scale: (float) confidence interval scale.
        inv_depth: (bool) whether to inverse depth (disparity).
        
    Returns:
        depth_map: (torch.Tensor) shape (B, 1, H, W), predicted depth map.
        ci: (torch.Tensor) shape (B, 2, H, W), predicted depth confidence interval.
    """ 
    depth = torch.sum(depth_prob * depth_values, dim=1, keepdim=True)  # (B, 1, H, W)
    var =  torch.sum(depth_prob * (depth_values - depth).square(), dim=1, keepdim=True)  # (B, 1, H, W)
    std = torch.clamp_min(var, 1e-12).sqrt()  # (B, 1, H, W)
    # std = var.sqrt()  # (B, 1, H, W)
        
    half_ci = ci_scale * std  # (B, 1, H, W)
        
    if inv_depth:  # to depth
        ci = 1. / torch.cat((torch.min(depth + half_ci, depth_values[:, :1]), 
                             torch.max(depth - half_ci, depth_values[:, -1:])), 
                            dim=1)  # (B, 2, H, W)
        # ci = 1. / torch.cat((depth + half_ci, depth - half_ci), dim=1)  # (B, 2, H, W)
        depth = 1. / depth
    else:
        ci = torch.cat((torch.max(depth - half_ci, depth_values[:, :1]), 
                        torch.min(depth + half_ci, depth_values[:, -1:])), 
                       dim=1)  # (B, 2, H, W)
        # ci = torch.cat((depth - half_ci, depth + half_ci), dim=1)  # (B, 2, H, W)
        
    return depth, ci

