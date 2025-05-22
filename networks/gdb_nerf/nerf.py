import torch
import torch.nn as nn
import torch.nn.functional as F


class NeRF(nn.Module):
    def __init__(self, 
                 hid_dim: int = 64, 
                 feat_dim: int = 16, 
                 voxel_dim: int = 8, 
                 viewdir_agg: bool = True) -> None:
        """Generalizable NeRF (MVSNeRF) based on agregated 2D, 3D features and directions.
        """
        super(NeRF, self).__init__()
        self.feat_dim = feat_dim
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
        self.weight = nn.Sequential(
            nn.Linear(hid_dim+voxel_dim+16+feat_dim+3+4, hid_dim), 
            nn.ReLU(inplace=True), 
            nn.Linear(hid_dim, 1), 
            nn.ReLU(inplace=True)
        )
        self.feat_head = nn.Sequential(
            nn.Linear(hid_dim, voxel_dim), 
            nn.ReLU(inplace=True)
        )
    
    def agg_viewdir(self, 
                    feat_rgb_dir: torch.Tensor) -> torch.Tensor:
        """Aggregate view direction features.
        Args:
            feat_rgb_dir: (torch.Tensor) shape (num_views, num_points, feat_dim + rgbs (3) + dir (4)).
        Returns:
            _: (torch.Tensor) shape (num_points, 16)
        """
        num_views = feat_rgb_dir.shape[0]
        img_feat_rgb = feat_rgb_dir[..., :-4]

        if self.viewdir_agg:
            view_feat = self.view_fc(feat_rgb_dir[..., -4:])  # (num_views, num_points, feat_dim+3)
            img_feat_rgb = img_feat_rgb + view_feat  # (num_views, num_points, feat_dim+3)

        var_feat, avg_feat = torch.var_mean(img_feat_rgb, dim=0, keepdim=True)
        var_feat = var_feat.expand(num_views, -1, -1)
        avg_feat = avg_feat.expand(num_views, -1, -1)

        feat = torch.cat([img_feat_rgb, var_feat, avg_feat], dim=-1)  # (num_views, num_points, (feat_dim+3)*3)
        global_feat = self.global_fc(feat)  # (num_views, num_points, 32)
        agg_w = F.softmax(self.agg_w_fc(global_feat), dim=0)  # (num_views, num_points, 1)
        im_feat = torch.sum(global_feat * agg_w, dim=0)  # (num_points, 32)
        
        return self.fc(im_feat)

    def forward(self, 
                vox_feat: torch.Tensor, 
                rgbs_feat_rgb_dir: torch.Tensor, 
                only_geo: bool = False) -> torch.Tensor:
        """Predict color/feat and sigma for each ray sample.
        Args:
            vox_feat: (torch.Tensor) shape (num_points, voxel_dim), voxel features from cost volume.
            rgbs_feat_rgb_dir: (torch.Tensor) shape (num_views, num_points, rgbs (3) * up_scale**2 + feat_dim + rgb (3) + dir (4)).
            only_geo: (bool) only predict sigma.
        Returns:
            sigma: (torch.Tensor) shape (num_points,).
            rgbs_feat_rgb: (torch.Tensor) shape (num_points, 3*up_scale**2+feat_dim+3).
        """
        num_views = rgbs_feat_rgb_dir.shape[0]
        feat_rgb_dir = rgbs_feat_rgb_dir[..., -(self.feat_dim+3+4):]  # (num_views, num_points, feat_dim+3+4)
        img_feat = self.agg_viewdir(feat_rgb_dir)  # (num_points, 16)
        vox_img_feat = torch.cat((vox_feat, img_feat), dim=-1)  # (num_points, voxel_dim+16)
        x = self.lr0(vox_img_feat)  # (num_points, hid_dim)
        sigma = self.sigma(x)  # (num_points, 1)

        rgbs_feat_rgb = None
        if not only_geo:
            w_feat = torch.cat((x, vox_img_feat), dim=-1)  # (num_points, hid_dim+voxel_dim+16)
            w_feat = w_feat.unsqueeze(0).expand(num_views, -1, -1)  # (num_views, num_points, hid_dim+voxel_dim+16)
            w_feat = torch.cat((w_feat, feat_rgb_dir), dim=-1)  # (num_views, num_points, hid_dim+voxel_dim+16+feat_dim+3+4)
            weight = F.softmax(self.weight(w_feat), dim=0)  # (num_views, num_points, 1)
            rgbs_feat_rgb = torch.sum((rgbs_feat_rgb_dir[..., :-4] * weight), dim=0)  # (num_points, 3*up_scale**2+feat_dim+3)
            feat = torch.cat((rgbs_feat_rgb, 
                              self.feat_head(x)), 
                              dim=-1)  # (num_points, 3*up_scale**2+feat_dim+3+voxel_dim)

        return sigma.squeeze(-1), feat
