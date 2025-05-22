import math
import torch
import torch.nn.functional as F
import nvdiffrast.torch
from typing import Union, List, Tuple


class BundleSampler:
    def __init__(self,
                 global_num_depth: int, 
                 max_mipmap_level: int) -> None:
        """Bundle sampler for NeRF."""
        super(BundleSampler, self).__init__()

        # Hyperparameters
        self.global_num_depth = global_num_depth  # number of global samples per ray
        self.max_mipmap_level = max_mipmap_level  # maximum mipmap level to use for interpolation

        # Original rays
        self.H_orig = None  # original image height
        self.W_orig = None  # original image width
        self.rays_o = None  # (B, 3), ray origins
        self.rays_d = None  # (B, H_orig, W_orig, 3), unnormalized ray directions
        self.uv = None  # (H_orig, W_orig, 2), normalized pixel grid
        self.tar_pixel_radius = None  # (B,), target pixel radius
        self.z_axis = None  # (B, 3), z-axis in world coordinates
        self.near = None  # (B,), near depth of the scene
        self.far = None  # (B,), far depth of the scene
    
    def build_rays(self,
                   tar_exts: torch.Tensor, 
                   tar_ints: torch.Tensor, 
                   im_size: Union[Tuple[int, int], List[int]], 
                   near: torch.Tensor, 
                   far: torch.Tensor) -> None:
        """Builds rays for NeRF.
        
        Args:
            tar_exts: (torch.Tensor) shape (B, 4, 4), target camera extrinsics (w2c).
            tar_ints: (torch.Tensor) shape (B, 3, 3), target camera intrinsics.
            im_size: (Union[Tuple[int, int], List[int]]) target image size (H_orig, W_orig).
            near: (torch.Tensor) shape (B,), near depth of the scene.
            far: (torch.Tensor) shape (B,), far depth of the scene.
        """
        device = tar_exts.device
        dtype = tar_exts.dtype
        B = tar_exts.shape[0]
        self.H_orig, self.W_orig = im_size
        self.near = near
        self.far = far
        
        # Compute pixel grid
        x, y = torch.meshgrid(torch.arange(self.W_orig, dtype=dtype, device=device) + 0.5, 
                              torch.arange(self.H_orig, dtype=dtype, device=device) + 0.5, 
                              indexing='xy')  # (H_orig, W_orig)
        self.uv = torch.stack((2 * x / self.W_orig - 1, 2 * y / self.H_orig - 1), dim=-1)  # (H_orig, W_orig, 2), scaled to [-1, 1]
        
        xyz = torch.stack(
            (
                x.flatten(), 
                y.flatten(), 
                torch.ones(self.H_orig * self.W_orig, dtype=dtype, device=device)
            ), dim=1
        )  # (H_orig * W_orig, 3)
        
        # To world coordinates
        c2w = torch.inverse(tar_exts)  # (B, 4, 4)
        self.z_axis = c2w[:, :3, 2]  # (B, 3), z-axis in world coordinates
        self.rays_o = c2w[:, :3, 3]  # (B, 3)
        rays_d = torch.matmul(xyz, torch.matmul(c2w[:, :3, :3], torch.inverse(tar_ints)).transpose(-2, -1))  # (B, H_orig * W_orig, 3)
        self.rays_d = rays_d.view(B, self.H_orig, self.W_orig, 3)  # (B, H_orig, W_orig, 3)
        
        # Compute pixel radius in the target view
        self.tar_pixel_radius = 1. / torch.sqrt(tar_ints[:, 0, 0] * tar_ints[:, 1, 1] * torch.pi)  # (B,)

    def _assemble_bundles(self, 
                          depth_range: torch.Tensor, 
                          vol_range: torch.Tensor, 
                          b_size: int) -> torch.Tensor:
        """Assemble bundles for NeRF.
        
        Args:
            depth_range: (torch.Tensor) shape (B, 2, H, W), depth/disparity range of bundles.
            vol_range: (torch.Tensor) shape (B, 2, H, W), depth/disparity range of depth hypothesis.
            b_size: (int) bundle size.
            
        Returns:
            bundles: (torch.Tensor) shape (B, H, W, 11+3*b_size**2), rays origin (3), unnormalized ray directions (3*b_size**2), 
                normalized pixel coordinates (2), bundle depth range (2), volume depth range (2), bundle disk radii (1), and cos 
                of bundle angle with z-axis (1).
        """
        rays_o = self.rays_o  # (B, 3)
        rays_d = self.rays_d  # (B, H_orig, W_orig, 3)
        uv = self.uv  # (H_orig, W_orig, 2)
        B = rays_d.shape[0]
        H, W = self.H_orig // b_size, self.W_orig // b_size  # spatial size of the bundle map
        
        rays_d = rays_d.view(B, H, b_size, W, b_size, 3)
        bundle_d = rays_d.mean(dim=(2, 4))  # (B, H, W, 3)
        rays_d = rays_d.permute(0, 1, 3, 5, 2, 4).reshape(B, H, W, 3 * b_size * b_size)
        z_axis = self.z_axis.unsqueeze(1).unsqueeze(1)  # (B, 1, 1, 3), z-axis in world coordinates
        bundle_cos = torch.sum(bundle_d * z_axis, dim=-1, keepdim=True) / torch.linalg.vector_norm(bundle_d, dim=-1, keepdim=True)  # (B, H, W, 1)

        uv = uv.view(H, b_size, W, b_size, 2).mean(dim=(1, 3))  # (H, W, 2), scaled to [-1, 1]
        uv = uv.unsqueeze(0).expand(B, -1, -1, -1)  # (B, H, W, 2)
        bundle_disk_radii = (b_size * self.tar_pixel_radius).view(B, 1, 1, 1).expand(-1, H, W, -1)  # (B, H, W, 1)
        
        # Assemble bundles
        bundles = torch.cat(
            (
                rays_o.view(B, 1, 1, 3).expand(-1, H, W, -1),  # (B, H, W, 3)
                rays_d,  # (B, H, W, 3*b_size**2)
                uv,  # (B, H, W, 2)
                depth_range.permute(0, 2, 3, 1),  # (B, H, W, 2)
                vol_range.permute(0, 2, 3, 1),  # (B, H, W, 2)
                bundle_disk_radii,  # (B, H, W, 1)
                bundle_cos  # (B, H, W, 1)
            ), dim=-1
        )
        return bundles
    
    def _sample_along_depth(self, 
                            bundle_near: torch.Tensor, 
                            bundle_far: torch.Tensor, 
                            num_samples: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample along bundles at equal depth intervals, total_num_samples = sum(samples_per_bundle).
        
        Args:
            bundle_near: (torch.Tensor) shape (num_bundles, 1), near depth/disparity of bundles.
            bundle_far: (torch.Tensor) shape (num_bundles, 1), far depth/disparity of bundles.
            num_samples: (int) number of samples per bundle.

        Returns:
            bundle_indices: (torch.Tensor) shape (total_num_samples,), bundle indices of the samples.
            t_starts: (torch.Tensor) shape (total_num_samples,), starting depth/disparity value of each sampled bin.
            t_ends: (torch.Tensor) shape (total_num_samples,), ending depth/disparity value of each sampled bin.
            samples_per_bundle: (torch.Tensor) shape (num_bundles,), number of samples per bundle.
        """
        device = bundle_near.device
        num_bundles = bundle_near.shape[0]
        
        # Ray marching
        indices = torch.arange(num_samples+1, device=device).unsqueeze(0).expand(num_bundles, -1)  # (num_bundles, num_samples+1)
        t_vals = bundle_near + (bundle_far - bundle_near) / num_samples * indices  # (num_bundles, num_samples+1)
        t_starts = t_vals[:, :-1].flatten()  # (total_num_samples,)
        t_ends = t_vals[:, 1:].flatten()  # (total_num_samples,)
        
        # bundle_indices indicates which sample belongs to which bundle
        bundle_indices = torch.arange(num_bundles, device=device)[:, None].expand(-1, num_samples)  # (num_bundles, num_samples)
        bundle_indices = bundle_indices.flatten()  # (total_num_samples,)
        
        samples_per_bundle = torch.full((num_bundles,), num_samples, dtype=torch.int32, device=device)
        
        return bundle_indices, t_starts, t_ends, samples_per_bundle
    
    def _sample_along_depth_adaptive(self, 
                                     bundle_near: torch.Tensor, 
                                     bundle_far: torch.Tensor, 
                                     min_sample_interval: torch.Tensor, 
                                     max_num_samples: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample along bundles at equal depth intervals, total_num_samples = sum(samples_per_bundle).
        
        Args:
            bundle_near: (torch.Tensor) shape (num_bundles, 1), near depth/disparity of bundles.
            bundle_far: (torch.Tensor) shape (num_bundles, 1), far depth/disparity of bundles.
            min_sample_interval: (torch.Tensor) shape (num_bundles, 1), minimum sampling interval.
            max_num_samples: (int) maximum number of samples per bundle.

        Returns:
            bundle_indices: (torch.Tensor) shape (total_num_samples,), bundle indices of the samples.
            t_starts: (torch.Tensor) shape (total_num_samples,), starting depth/disparity value of each sampled bin.
            t_ends: (torch.Tensor) shape (total_num_samples,), ending depth/disparity value of each sampled bin.
            samples_per_bundle: (torch.Tensor) shape (num_bundles,), number of samples per bundle.
        """
        device = bundle_near.device
        num_bundles = bundle_near.shape[0]
        
        # Adaptive ray marching
        samples_per_bundle = torch.ceil(torch.abs(bundle_far - bundle_near) / min_sample_interval).clamp(1, max_num_samples)  # (num_bundles, 1)
        indices = torch.arange(max_num_samples+1, device=device).unsqueeze(0).expand(num_bundles, -1)  # (num_bundles, max_num_samples+1)
        
        valid = indices[:, :-1] < samples_per_bundle  # (num_bundles, max_num_samples)
        t_vals = bundle_near + (bundle_far - bundle_near) / samples_per_bundle * indices  # (num_bundles, max_num_samples+1)
        t_starts = t_vals[:, :-1][valid]  # (total_num_samples,)
        t_ends = t_vals[:, 1:][valid]  # (total_num_samples,)
        
        # bundle_indices indicates which sample belongs to which bundle
        bundle_indices = torch.arange(num_bundles, device=device).unsqueeze(1).expand(-1, max_num_samples)  # (num_bundles, max_num_samples)
        bundle_indices = bundle_indices[valid]  # (total_num_samples,)
        
        return bundle_indices, t_starts, t_ends, samples_per_bundle.squeeze(-1)
    
    def sample(self, 
               depth_range: torch.Tensor, 
               vol_range: torch.Tensor, 
               b_size: int, 
               max_num_samples: int, 
               inv_depth: bool = False, 
               is_adaptive: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Samples bundles and generates soft/binary mask for NeRF.
        
        Args:
            depth_range: (torch.Tensor) shape (B, 2, H, W), depth range of bundles.
            vol_range: (torch.Tensor) shape (B, 2, H, W), depth range of depth hypothesis.
            b_size: (int) bundle size. 
            max_num_samples: (int) maximum number of samples per bundle.
            inv_depth: (bool) whether to use inverse depth for sampling.
            is_adaptive: (bool) whether to use adaptive sampling.
            
        Returns:
            rays_xyz: (torch.Tensor) shape (total_num_samples, 3, b_size**2), 3D world coordinates of ray samples.
            uvd: (torch.Tensor) shape (total_num_samples, 3), normalized 3D grid coordinates of sphere centers, [-1, 1].
            z_vals: (torch.Tensor) shape (total_num_samples,), depth value of sphere centers.
            ball_radii: (torch.Tensor) shape (total_num_samples,), ball radii of sampled spheres.
            indices: (torch.Tensor) shape (total_num_samples,), indices of the sampled bundles.
            samples_per_batch: (torch.Tensor) shape (B,), number of samples per batch.
            samples_per_bundle: (torch.Tensor) shape (num_bundles,), number of samples per bundle.
        """
        B, _, H, W = depth_range.shape
        if self.rays_o is None:
            raise ValueError("Rays have not been built yet. Please call build_rays() first.")
        
        # Assemble bundles
        if inv_depth:  # to disparity
            depth_range = 1. / depth_range
            vol_range = 1. / vol_range
            min_sample_interval = (1. / self.near - 1. / self.far) / self.global_num_depth  # (B,), minimum sampling interval
        else:
            min_sample_interval = (self.far - self.near) / self.global_num_depth  # (B,)
        bundles = self._assemble_bundles(depth_range, vol_range, b_size)  # (B, H, W, 11+3*b_size**2)
        bundles = bundles.flatten(0, 2)  # (B*H*W, 11+3*b_size**2)
        min_sample_interval = min_sample_interval.unsqueeze(-1).expand(-1, H * W).reshape(-1, 1)  # (B*H*W, 1)

        # Depth guided raymarching, total_num_samples = sum(samples_per_bundle)
        bundle_near, bundle_far = bundles[:, -6:-5], bundles[:, -5:-4]
        if is_adaptive:
            indices, t_starts, t_ends, samples_per_bundle = self._sample_along_depth_adaptive(bundle_near, bundle_far, min_sample_interval, max_num_samples)  # (total_num_samples,)
        else:
            indices, t_starts, t_ends, samples_per_bundle = self._sample_along_depth(bundle_near, bundle_far, max_num_samples)
        
        # Sphere-based sampling
        samples_per_batch = torch.sum(samples_per_bundle.view(B, -1), dim=1)  # (B,)
        total_num_samples = indices.shape[0]
        rays_o, rays_d, uv = bundles[indices, :3], bundles[indices, 3:-8], bundles[indices, -8:-6]  # (total_num_samples, _)
        vol_near, vol_far = bundles[indices, -4], bundles[indices, -3]  # (total_num_samples,)
        z_vals = 0.5 * (t_starts + t_ends)  # (total_num_samples,), depth/disparity value of the midpoint of each sampling interval
        d = 2 * (z_vals - vol_near) / (vol_far - vol_near) - 1.  # z-axis value, scaled to [-1, 1]
        uvd = torch.cat([uv, d.unsqueeze(1)], dim=-1)  # (total_num_samples, 3), normalized 3D grid coordinates of samples, [-1, 1]

        if inv_depth:  # to depth
            z_vals = 1. / z_vals
        
        # Calculate the positions of ray samples
        rays_d = rays_d.view(total_num_samples, 3, -1)  # (total_num_samples, 3, b_size**2)
        rays_xyz = rays_o.unsqueeze(-1) + rays_d * z_vals[:, None, None]  # (total_num_samples, 3, b_size**2), 3D world coordinates of ray samples
        bundle_xyz = rays_xyz.mean(dim=-1)  # (total_num_samples, 3), 3D world coordinates of bundle samples' centers

        # Compute ball radii of sampled samples
        distances = torch.linalg.vector_norm(bundle_xyz - rays_o, dim=-1)  # (total_num_samples,)
        bundle_disk_radii = bundles[:, -2]  # (B*H*W,)
        bundle_cos = bundles[:, -1]  # cos of the angle between the axis of the bundle cone and the z-axis, (B*H*W,)
        ball_radii = bundle_disk_radii * bundle_cos / torch.sqrt((torch.sqrt((1. / bundle_cos.square() - 1.).clamp_min(1e-12)) - bundle_disk_radii).square() + 1.)  # (B*H*W,)
        ball_radii = distances * ball_radii[indices]  # (total_num_samples,)

        return rays_xyz, uvd, z_vals, ball_radii, indices, samples_per_batch, samples_per_bundle
    
    def encode(self, 
               src_images: torch.Tensor, 
               img_feat: torch.Tensor, 
               feat_volume: torch.Tensor, 
               rays_xyz: torch.Tensor, 
               uvd: torch.Tensor, 
               ball_radii: torch.Tensor, 
               src_exts: torch.Tensor, 
               src_ints: torch.Tensor, 
               tar_exts: torch.Tensor, 
               samples_per_batch: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode samples using features from the source image.
        
        Args:
            src_images: (torch.Tensor) shape (B, num_views, 3, H_orig, W_orig), source images.
            img_feat: (torch.Tensor) shape (B, num_views, feat_dim, H, W), source feature maps.
            feat_volume: (torch.Tensor) shape (B, voxel_dim, num_depth, H, W), 3D feature volume.
            rays_xyz: (torch.Tensor) shape (total_num_samples, 3, b_size*b_size), 3D world coordinates of ray samples.
            uvd: (torch.Tensor) shape (total_num_samples, 3), normalized 3D grid coordinates of sphere centers, [-1, 1].
            ball_radii: (torch.Tensor) shape (total_num_samples,), ball radii of sampled spheres.
            src_exts: (torch.Tensor) shape (B, num_views, 4, 4), source extrinsic matrices (w2c).
            src_ints: (torch.Tensor) shape (B, num_views, 3, 3), source intrinsic matrices.
            tar_exts: (torch.Tensor) shape (B, 4, 4), target extrinsic matrix (w2c).
            samples_per_batch: (torch.Tensor) shape (B,), number of samples per batch.
        
        Returns:
            rgbs_feat_dir: (torch.Tensor) shape (num_views, total_num_samples, 3*b_size*b_size+feat_dim+4), rgbs (3*b_size*b_size) 
                + image feature (feat_dim) + dir (4).
            vox_feat: (torch.Tensor) shape (total_num_samples, voxel_dim), 3D feature of each sample.
        """
        dtype=img_feat.dtype
        device=img_feat.device
        B, num_views, feat_dim, H, W = img_feat.shape
        total_num_samples, _, b_size_sq = rays_xyz.shape  # b_size_sq = b_size * b_size
        b_size = round(math.sqrt(b_size_sq))  # b_size

        # Compute camera centers in the world coordinates
        tar_cam_xyz = torch.inverse(tar_exts)[:, None, :3, 3]  # (B, 1, 3)
        src_cam_xyz = torch.inverse(src_exts)[..., :3, 3]  # (B, num_views, 3)

        # Compute sphere centers in the world coordinates
        bundle_xyz = rays_xyz.mean(dim=-1)  # (total_num_samples, 3)

        # Compute the pixel radius of the source view
        src_ints_scaled = src_ints.clone()
        src_ints_scaled[..., :2, :] = src_ints_scaled[..., :2, :] / b_size
        src_pixel_radii = 1 / torch.sqrt(src_ints_scaled[:, :, 0, 0] * src_ints_scaled[:, :, 1, 1] * torch.pi)  # (B, num_views)

        start_idx = 0
        vox_feat = torch.empty((total_num_samples, feat_volume.shape[1]), dtype=dtype, device=device)
        rgbs_feat_dir = torch.empty((num_views, total_num_samples, 3*b_size_sq+feat_dim+4), dtype=dtype, device=device)
        for b in range(B):  # loop over batches
            num_samples = int(samples_per_batch[b])
            
            # Extract the voxel-aligned feature from the 3D feature volume, following ENeRF
            sub_uvd = uvd[None, start_idx:start_idx + num_samples, None, None]  # (1, num_samples, 1, 1, 3)
            sub_vox_feat = F.grid_sample(feat_volume[b:b+1], sub_uvd, mode='bilinear', padding_mode='border', align_corners=False)  # (1, voxel_dim, num_samples, 1, 1)
            vox_feat[start_idx:start_idx + num_samples] = sub_vox_feat.view(-1, num_samples).permute(1, 0)

            # Compute samples in camera coordinates
            rays_xyz_cam = rays_xyz[start_idx:start_idx+num_samples].permute(0, 2, 1).reshape(1, -1, 3)  # (1, num_samples*b_size*b_size, 3)
            rays_xyz_cam = F.pad(rays_xyz_cam, (0, 1), mode='constant', value=1)  # (1, num_samples*b_size*b_size, 4)
            rays_xyz_cam = torch.matmul(rays_xyz_cam, src_exts[b].transpose(-2, -1))[..., :3]  # (num_views, num_samples*b_size*b_size, 3)

            # Compute samples in image coordinates
            rays_xyz_img = torch.matmul(rays_xyz_cam, src_ints[b].transpose(-2, -1))  # (num_views, num_samples*b_size*b_size, 3)
            rays_grid = rays_xyz_img[..., :2] / rays_xyz_img[..., 2:3].clamp_min(1e-6)  # (num_views, num_samples*b_size*b_size, 2)
            rays_grid[..., 0], rays_grid[..., 1] = 2 * rays_grid[..., 0] / self.W_orig - 1., 2 * rays_grid[..., 1] / self.H_orig - 1.
            
            rgbs = F.grid_sample(src_images[b], rays_grid.unsqueeze(2), align_corners=False, mode='bilinear', padding_mode='border')  # (num_views, 3, num_samples*b_size*b_size, 1)
            rgbs = rgbs.view(num_views, 3, -1, b_size_sq).permute(0, 2, 1, 3).reshape(num_views, num_samples, 3 * b_size_sq)
            
            # Compute sphere centers in camera coordinates
            bundle_xyz_cam = rays_xyz_cam.view(num_views, -1, b_size_sq, 3).mean(dim=-2)  # (num_views, num_samples, 3)
            
            # Compute the radii of sphere projections in the target view
            distances = torch.linalg.vector_norm(bundle_xyz_cam, dim=-1, keepdim=True)  # (num_views, num_samples, 1)
            sec_sq = (distances / bundle_xyz_cam[..., 2:3]).square()  # squared sec of the angle between the bundle axis and the z-axis, (num_views, num_samplessec_sq, 1)

            proj_disk_radii = sec_sq / (torch.sqrt(((distances / ball_radii[None, start_idx:start_idx+num_samples, None]).square() - 1.).clamp_min(1e-12)) \
                                        + torch.sqrt((sec_sq - 1.).clamp_min(1e-12)))  # (num_views, num_samples, 1)
            levels = torch.log2(proj_disk_radii / src_pixel_radii[b, :, None, None])  # (num_views, num_samples, 1)
            
            # Project sphere centers to source images
            bundle_xyz_img = torch.matmul(bundle_xyz_cam, src_ints_scaled[b].transpose(-2, -1))  # (num_views, num_samples, 3)
            bundle_grid = bundle_xyz_img[..., :2] / bundle_xyz_img[..., 2:3].clamp_min(1e-6)  # (num_views, num_samples, 2)
            bundle_grid[..., 0], bundle_grid[..., 1] = bundle_grid[..., 0] / W, bundle_grid[..., 1] / H
            
            feat = nvdiffrast.torch.texture(img_feat[b].permute(0, 2, 3, 1).contiguous(), # (num_views, H, W, feat_dim)
                                            bundle_grid.unsqueeze(2).contiguous(), # (num_views, num_samples, 1, 2)
                                            mip_level_bias=levels.contiguous(), # (num_views, num_samples, 1)
                                            boundary_mode='clamp', 
                                            max_mip_level=self.max_mipmap_level).squeeze(2)  # (num_views, num_samples, feat_dim)
            
            # Compute the difference between the rays for each sphere center in the source and target views
            tar_diff = F.normalize(bundle_xyz[start_idx:start_idx+num_samples] - tar_cam_xyz[b], p=2.0, dim=-1)  # (num_samples, 3)
            src_diff = F.normalize(bundle_xyz[start_idx:start_idx+num_samples] - src_cam_xyz[b].unsqueeze(1), p=2.0, dim=-1)  # (num_views, num_samples, 3)
            
            ray_diff_direction = F.normalize(tar_diff - src_diff, p=2.0, dim=-1)  # (num_views, num_samples, 3)
            ray_diff_dot = torch.sum(tar_diff * src_diff, dim=-1, keepdim=True)  # (num_views, num_samples, 1)
            ray_diff = torch.cat([ray_diff_direction, ray_diff_dot], dim=-1)  # (num_views, num_samples, 4)

            rgbs_feat_dir[:, start_idx:start_idx+num_samples] = torch.cat([rgbs, feat, ray_diff], dim=-1)  # (num_views, num_samples, 3*b_size*b_size+feat_dim+4)
            start_idx += num_samples
        return rgbs_feat_dir, vox_feat