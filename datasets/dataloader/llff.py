import os
import random
import cv2
import torch
import numpy as np


class Dataset:
    def __init__(self, cfg, **kwargs):
        super(Dataset, self).__init__()
        self.cfg = cfg
        self.data_root = os.path.join(self.cfg.workspace, kwargs['data_root'])
        self.split = kwargs['split']
        self.input_h_w = kwargs['input_h_w']
        if 'scene' in kwargs:
            self.scenes = [kwargs['scene']]
        else:
            self.scenes = []
        self.build_metas()

    def build_metas(self):
        if len(self.scenes) == 0:
            scenes = ['fern', 'flower', 'fortress', 'horns', 'leaves', 'orchids', 'room', 'trex']
        else:
            scenes = self.scenes
        self.scene_infos = {}
        self.metas = []
        pairs = torch.load('data/mvsnerf/pairs.th')
        for scene in scenes:
            pose_bounds = np.load(os.path.join(self.data_root, scene, 'poses_bounds.npy'))  # c2w, -u, r, -t
            poses = pose_bounds[:, :15].reshape((-1, 3, 5))
            c2ws = np.eye(4)[None].repeat(len(poses), 0)
            c2ws[:, :3, 0], c2ws[:, :3, 1], c2ws[:, :3, 2], c2ws[:, :3, 3] = poses[:, :3, 1], poses[:, :3, 0], -poses[:, :3, 2], poses[:, :3, 3]
            ixts = np.eye(3)[None].repeat(len(poses), 0)
            ixts[:, 0, 0], ixts[:, 1, 1] = poses[:, 2, 4], poses[:, 2, 4]
            ixts[:, 0, 2], ixts[:, 1, 2] = poses[:, 1, 4]/2., poses[:, 0, 4]/2.
            ixts[:, :2] *= 0.25

            img_paths = sorted([item for item in os.listdir(os.path.join(self.data_root, scene, 'images_4')) if '.png' in item])
            depth_ranges = pose_bounds[:, -2:]
            scene_info = {'ixts': ixts.astype(np.float32), 'c2ws': c2ws.astype(np.float32), 'image_names': img_paths, 'depth_ranges': depth_ranges.astype(np.float32)}
            scene_info['scene_name'] = scene
            self.scene_infos[scene] = scene_info

            train_ids = pairs[f'{scene}_train']
            if self.split == 'train':
                render_ids = train_ids
            else:
                render_ids = pairs[f'{scene}_val']

            c2ws = c2ws[train_ids]
            for i in render_ids:
                c2w = scene_info['c2ws'][i]
                distance = np.linalg.norm((c2w[:3, 3][None] - c2ws[:, :3, 3]), axis=-1)
                argsorts = distance.argsort()
                argsorts = argsorts[1:] if i in train_ids else argsorts
                input_views_num = max(self.cfg.train.sampler_meta.input_views_num) if self.split == 'train' else self.cfg.test.sampler_meta.input_views_num[0]
                src_views = [train_ids[i] for i in argsorts[:input_views_num]]
                self.metas += [(scene, i, src_views)]

    def __getitem__(self, index_meta):
        index, input_views_num, render_scale = index_meta
        scene, tar_view, src_views = self.metas[index]
        if self.split == 'train':
            if np.random.random() < 0.1:
                src_views = src_views + [tar_view]
            src_views = random.sample(src_views, input_views_num)
        scene_info = self.scene_infos[scene]
        tar_img, tar_mask, tar_ext, tar_ixt = self.read_tar(scene_info, tar_view)
        src_inps, src_exts, src_ixts = self.read_src(scene_info, src_views)

        tar_gt_ms = {'rgb': [], 
                     'mask': []}
        
        # Multi-scale GT for depth estimation
        for s in self.cfg.mvs.vol_scales:
            tar_img_i = cv2.resize(tar_img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
            tar_msk_i = cv2.resize(tar_mask, None, fx=s, fy=s, interpolation=cv2.INTER_NEAREST)
            tar_gt_ms['rgb'].append(tar_img_i)
            tar_gt_ms['mask'].append(tar_msk_i)
        
        if render_scale != 1.0:
            tar_img = cv2.resize(tar_img, None, fx=render_scale, fy=render_scale, interpolation=cv2.INTER_AREA)
            tar_mask = cv2.resize(tar_mask, None, fx=render_scale, fy=render_scale, interpolation=cv2.INTER_NEAREST)
        
        src_views = {'rgb': src_inps, 
                     'extrinsics': src_exts, 
                     'intrinsics': src_ixts}
        tar_views = {'extrinsics': tar_ext, 
                     'intrinsics': tar_ixt, 
                     'rgb': tar_img, 
                     'mask': tar_mask}
        
        depth_ranges = np.array(scene_info['depth_ranges'])
        near_far = np.array([depth_ranges[:, 0].min().item(), depth_ranges[:, 1].max().item()], dtype=np.float32)

        H, W = tar_img.shape[:2]
        meta = {'scene': scene, 'tar_view': tar_view, 'frame_id': 0, 'h': H, 'w': W}

        return {'src_views': src_views, 
                'tar_views': tar_views, 
                'near_far': near_far, 
                'tar_gt_ms': tar_gt_ms, 
                'render_scale': render_scale, 
                'meta': meta}

    def read_src(self, scene, src_views):
        src_ids = src_views
        ixts, exts, imgs = [], [], []
        for idx in src_ids:
            img, orig_size = self.read_image(scene, idx)
            imgs.append(img)
            ixt, ext = self.read_cam(scene, idx, orig_size)
            ixts.append(ixt)
            exts.append(ext)
        return np.stack(imgs).transpose((0, 3, 1, 2)), np.stack(exts), np.stack(ixts)

    def read_tar(self, scene, view_idx):
        img, orig_size = self.read_image(scene, view_idx)
        ixt, ext = self.read_cam(scene, view_idx, orig_size)
        mask = np.ones_like(img[..., 0], dtype=np.float32)
        return img, mask, ext, ixt

    def read_cam(self, scene, view_idx, orig_size):
        c2w = scene['c2ws'][view_idx]
        w2c = np.linalg.inv(c2w)
        ixt = scene['ixts'][view_idx].copy()
        ixt[0] *= self.input_h_w[1] / orig_size[1]
        ixt[1] *= self.input_h_w[0] / orig_size[0]
        return ixt, w2c

    def read_image(self, scene, view_idx):
        image_path = os.path.join(self.data_root, scene['scene_name'], 'images_4', scene['image_names'][view_idx])
        img = cv2.cvtColor(cv2.imread(image_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.
        orig_size = img.shape[:2]
        img = cv2.resize(img, self.input_h_w[::-1], interpolation=cv2.INTER_AREA)
        return img, orig_size

    def __len__(self):
        return len(self.metas)

