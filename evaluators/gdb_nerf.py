import os
import cv2
import torch
import lpips
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from utils import img_utils
from matplotlib import cm


class Evaluator:
    def __init__(self, cfg):
        self.cfg = cfg
        self.psnrs = []
        self.ssims = []
        self.lpips = []
        self.scene_psnrs = {}
        self.scene_ssims = {}
        self.scene_lpips = {}
        self.loss_fn_vgg = lpips.LPIPS(net='vgg')
        self.loss_fn_vgg.cuda()
        if self.cfg.test.eval_depth:
            # Following the setup of MVSNeRF
            self.eval_depth_scenes = ['scan1', 'scan8', 'scan21', 'scan103', 'scan110']
            self.abs = []
            self.acc_2 = []
            self.acc_10 = []
            self.mvs_abs = []
            self.mvs_acc_2 = []
            self.mvs_acc_10 = []
        os.system('mkdir -p ' + self.cfg.result_dir)

    def evaluate(self, output, batch):
        B, _, _, H, W = batch['src_views']['rgb'].shape
        
        gt_rgb = batch['tar_views']['rgb'].detach().cpu().numpy()
        masks = (batch['tar_views']['mask'].cpu().numpy() >= 1).astype(np.uint8)
        pred_rgb = output['rgb'].permute(0, 2, 3, 1).detach().clamp(0., 1.).cpu().numpy()
        
        if self.cfg.test.eval_center:
            H_crop, W_crop = int(H*0.1), int(W*0.1)
            pred_rgb = pred_rgb[:, H_crop:-H_crop, W_crop:-W_crop]
            gt_rgb = gt_rgb[:, H_crop:-H_crop, W_crop:-W_crop]
            masks = masks[:, H_crop:-H_crop, W_crop:-W_crop]

        for b in range(B):
            if not batch['meta']['scene'][b] in self.scene_psnrs:
                self.scene_psnrs[batch['meta']['scene'][b]] = []
                self.scene_ssims[batch['meta']['scene'][b]] = []
                self.scene_lpips[batch['meta']['scene'][b]] = []
            if self.cfg.save_result:
                # img = img_utils.horizon_concate(gt_rgb[b], pred_rgb[b])
                img_path = os.path.join(self.cfg.result_dir, '{}_{}_{}.png'.format(batch['meta']['scene'][b], batch['meta']['tar_view'][b].item(), batch['meta']['frame_id'][b].item()))
                # cv2.imwrite(img_path, (cv2.cvtColor(img, cv2.COLOR_RGB2BGR) * 255).clip(0, 255).astype(np.uint8))
                img = (cv2.cvtColor(pred_rgb[b], cv2.COLOR_RGB2BGR) * 255).clip(0, 255).astype(np.uint8)
                cv2.imwrite(img_path, img)

                # # save error map
                # gt_gray = cv2.cvtColor(gt_rgb[b], cv2.COLOR_RGB2GRAY)
                # pred_gray = cv2.cvtColor(pred_rgb[b], cv2.COLOR_RGB2GRAY)
                # error_map = np.abs(gt_gray - pred_gray)
                # error_map = cv2.normalize(error_map, None, 0, 1, cv2.NORM_MINMAX)
                # # mask = masks[b] == 1
                # # error_map[mask==False] = 0.
                # colormap = cm.jet_r(error_map)
                # colormap = (colormap[:, :, :3] * 255).astype(np.uint8)
                # cv2.imwrite(img_path.replace('.png', '_error.png'), colormap)

                # # save sampling map
                # sampling_map = output['samples_per_ray'][b].detach().cpu().numpy()
                # sampling_map = sampling_map[H_crop//2:-H_crop//2, W_crop//2:-W_crop//2]
                # sampling_map = cv2.normalize(sampling_map, None, 0, 1, cv2.NORM_MINMAX)
                # colormap = cm.jet_r(sampling_map)
                # colormap = (colormap[:, :, :3] * 255).astype(np.uint8)
                # cv2.imwrite(img_path.replace('.png', '_smpl.png'), colormap)

            mask = masks[b] == 1
            gt_rgb[b][mask==False] = 0.
            pred_rgb[b][mask==False] = 0.

            psnr_item = psnr(gt_rgb[b][mask], pred_rgb[b][mask], data_range=1.)
            self.psnrs.append(psnr_item)
            self.scene_psnrs[batch['meta']['scene'][b]].append(psnr_item)

            ssim_item = ssim(gt_rgb[b], pred_rgb[b], channel_axis=-1)
            self.ssims.append(ssim_item)
            self.scene_ssims[batch['meta']['scene'][b]].append(ssim_item)

            if self.cfg.eval_lpips:
                gt, pred = torch.Tensor(gt_rgb[b])[None].permute(0, 3, 1, 2), torch.Tensor(pred_rgb[b])[None].permute(0, 3, 1, 2)
                gt, pred = (gt-0.5)*2., (pred-0.5)*2.
                lpips_item = self.loss_fn_vgg(gt.cuda(), pred.cuda()).item()
                self.lpips.append(lpips_item)
                self.scene_lpips[batch['meta']['scene'][b]].append(lpips_item)

            if self.cfg.test.eval_depth and batch['meta']['scene'][b] in self.eval_depth_scenes:
                nerf_depth = output['nerf_depth'].cpu().numpy()[b]
                nerf_gt_depth = batch['tar_views']['depth'].cpu().numpy()[b]
                mvs_depth = output['mvs_depth'].cpu().numpy()[b]
                mvs_gt_depth = batch['tar_gt_ms']['depth'][-1][b].cpu().numpy()
                # nerf_gt_depth = cv2.resize(nerf_gt_depth, nerf_depth.shape[-1:-3:-1] , interpolation=cv2.INTER_NEAREST)
                nerf_depth = cv2.resize(nerf_depth, nerf_gt_depth.shape[-1:-3:-1] , interpolation=cv2.INTER_LINEAR)
                
                # nerf_mask = np.logical_and(nerf_gt_depth > 425., nerf_gt_depth < 905.)
                # mvs_mask = np.logical_and(mvs_gt_depth > 425., mvs_gt_depth < 905.)
                nerf_mask = nerf_gt_depth != 0.
                mvs_mask = mvs_gt_depth != 0.
                self.abs.append(np.abs(nerf_depth[nerf_mask] - nerf_gt_depth[nerf_mask]).mean())
                self.acc_2.append((np.abs(nerf_depth[nerf_mask] - nerf_gt_depth[nerf_mask]) < 2).mean())
                self.acc_10.append((np.abs(nerf_depth[nerf_mask] - nerf_gt_depth[nerf_mask]) < 10).mean())
                self.mvs_abs.append((np.abs(mvs_depth[mvs_mask] - mvs_gt_depth[mvs_mask])).mean())
                self.mvs_acc_2.append((np.abs(mvs_depth[mvs_mask] - mvs_gt_depth[mvs_mask]) < 2.).mean())
                self.mvs_acc_10.append((np.abs(mvs_depth[mvs_mask] - mvs_gt_depth[mvs_mask]) < 10.).mean())

    def summarize(self):
        ret = {}
        ret.update({'psnr': np.mean(self.psnrs)})
        ret.update({'ssim': np.mean(self.ssims)})
        if self.cfg.eval_lpips:
            ret.update({'lpips': np.mean(self.lpips)})
        print('='*30)
        for scene in self.scene_psnrs:
            if self.cfg.eval_lpips:
                print(scene.ljust(16), 'psnr: {:.2f} ssim: {:.3f} lpips:{:.3f}'.format(np.mean(self.scene_psnrs[scene]), np.mean(self.scene_ssims[scene]), np.mean(self.scene_lpips[scene])))
            else:
                print(scene.ljust(16), 'psnr: {:.2f} ssim: {:.3f} '.format(np.mean(self.scene_psnrs[scene]), np.mean(self.scene_ssims[scene])))
        print('='*30)
        print(ret)
        if self.cfg.test.eval_depth:
            depth_ret = {}
            keys = ['abs', 'acc_2', 'acc_10']
            for key in keys:
                depth_ret[key] = np.mean(getattr(self, key))
                setattr(self, key, [])
            print(depth_ret)
            keys = ['mvs_abs', 'mvs_acc_2', 'mvs_acc_10']
            depth_ret = {}
            for key in keys:
                depth_ret[key] = np.mean(getattr(self, key))
                setattr(self, key, [])
            print(depth_ret)
        self.psnrs = []
        self.ssims = []
        self.lpips = []
        self.scene_psnrs = {}
        self.scene_ssims = {}
        self.scene_lpips = {}
        if self.cfg.save_result:
            print('Save visualization results to: {}'.format(self.cfg.result_dir))
        return ret
