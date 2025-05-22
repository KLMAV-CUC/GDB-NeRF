import torch
import torch.nn as nn
from .depth_loss import SmoothLoss
from .photometric_loss import PhotometricLoss


class NetworkWrapper(nn.Module):
    def __init__(self, net, cfg):
        super(NetworkWrapper, self).__init__()
        self.device = torch.device('cuda:{}'.format(cfg.local_rank))
        self.net = net
        self.photometric_loss = PhotometricLoss([1.0, 0.1, 0.05])  # [1.0, 0.05, 0.02]
        self.loss_weight = cfg.mvs.loss_weight
        self.depth_loss = SmoothLoss(cfg.mvs.vol_scales)  # only for monitoring

    def forward(self, batch):
        output, depth_prediction, blend_rgbs = self.net(batch)

        scalar_stats = {}

        # color loss
        rgb_gt = batch['tar_views']['rgb'].permute(0, 3, 1, 2)
        rgb_est = output['rgb']

        color_loss, mse_loss, ssim, perceptual_loss = self.photometric_loss(rgb_gt, rgb_est)
        
        scalar_stats.update({'mse_loss': mse_loss})

        with torch.no_grad():
            psnr = -10. * torch.log10(mse_loss + 1e-6)
        scalar_stats.update({'psnr': psnr, 'ssim': ssim, f'perceptual_loss': perceptual_loss})

        # depth loss
        depth_loss = 0.
        if self.training:
            depth_rgb_est = blend_rgbs
            depth_rgb_gt = [rgb.permute(0, 3, 1, 2) for rgb in batch['tar_gt_ms']['rgb']]
            for i in range(len(depth_rgb_est)):
                depth_loss_i, _, _, _ = self.photometric_loss(depth_rgb_gt[i], depth_rgb_est[i])
                depth_loss += self.loss_weight[i] * depth_loss_i
            scalar_stats.update({f'depth_loss': depth_loss})
        
        with torch.no_grad():
            if 'depth' in batch['tar_gt_ms']:
                depth_gt_ms = batch['tar_gt_ms']['depth']
                depth_est_ms = depth_prediction
                mask = batch['tar_gt_ms']['mask']
                _, loss_dict = self.depth_loss(depth_est_ms, depth_gt_ms, mask)
                scalar_stats.update(loss_dict)

        loss = color_loss + depth_loss
        scalar_stats.update({'loss': loss})
        image_stats = {}

        return output, loss, scalar_stats, image_stats
