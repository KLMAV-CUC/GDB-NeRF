import os
import random
import cv2
import torch
import json
import numpy as np


class Dataset:
    def __init__(self, cfg, **kwargs):
        super(Dataset, self).__init__()
        self.cfg = cfg
        self.data_root = os.path.join(self.cfg.workspace, kwargs['data_root'])
        self.split = kwargs['split']
        if 'scene' in kwargs:
            self.scenes = [kwargs['scene']]
        else:
            self.scenes = []
        self.build_metas()

    def build_metas(self):
        if len(self.scenes) == 0:
            scenes = ['chair', 'drums', 'ficus', 'hotdog', 'lego', 'materials', 'mic', 'ship']
        else:
            scenes = self.scenes
        self.scene_infos = {}
        self.metas = []
        pairs = torch.load('data/mvsnerf/pairs.th')
        for scene in scenes:
            json_info = json.load(open(os.path.join(self.data_root, scene,'transforms_train.json')))
            b2c = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
            scene_info = {'ixts': [], 'exts': [], 'img_paths': []}
            for idx in range(len(json_info['frames'])):
                c2w = np.array(json_info['frames'][idx]['transform_matrix'])
                c2w = c2w @ b2c
                ext = np.linalg.inv(c2w)
                ixt = np.eye(3)
                ixt[0][2], ixt[1][2] = 400., 400.
                focal = .5 * 800 / np.tan(.5 * json_info['camera_angle_x'])
                ixt[0][0], ixt[1][1] = focal, focal
                scene_info['ixts'].append(ixt.astype(np.float32))
                scene_info['exts'].append(ext.astype(np.float32))
                img_path = os.path.join(self.data_root, scene, 'train/r_{}.png'.format(idx))
                scene_info['img_paths'].append(img_path)
            self.scene_infos[scene] = scene_info
            train_ids, render_ids = pairs[f'{scene}_train'], pairs[f'{scene}_val']
            if self.split == 'train':
                render_ids = train_ids
            c2ws = np.stack([np.linalg.inv(scene_info['exts'][idx]) for idx in train_ids])
            for idx in render_ids:
                c2w = np.linalg.inv(scene_info['exts'][idx])
                distance = np.linalg.norm((c2w[:3, 3][None] - c2ws[:, :3, 3]), axis=-1)

                argsorts = distance.argsort()
                argsorts = argsorts[1:] if idx in train_ids else argsorts

                input_views_num = max(self.cfg.train.sampler_meta.input_views_num) if self.split == 'train' else self.cfg.test.sampler_meta.input_views_num[0]
                src_views = [train_ids[i] for i in argsorts[:input_views_num]]
                self.metas += [(scene, idx, src_views)]

    def __getitem__(self, index_meta):
        index, input_views_num, render_scale = index_meta
        scene, tar_view, src_views = self.metas[index]
        if self.split == 'train':
            if np.random.random() < 0.1:
                src_views = src_views + [tar_view]
            src_views = random.sample(src_views, input_views_num)
        scene_info = self.scene_infos[scene]
        scene_info['scene_name'] = scene
        tar_img, tar_mask, tar_ext, tar_ixt = self.read_tar(scene_info, tar_view)
        src_inps, src_exts, src_ixts = self.read_src(scene_info, src_views)

        tar_gt_ms = {'rgb': [], 
                     'mask': []}
        
        # Multi-scale GT for depth estimation
        for s in self.cfg.mvs.vol_scales:
            tar_img_i = cv2.resize(tar_img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
            tar_msk_i = cv2.resize(tar_mask, None, fx=s, fy=s, interpolation=cv2.INTER_NEAREST)
            tar_gt_ms['rgb'].append(tar_img_i.astype(np.float32))
            tar_gt_ms['mask'].append(tar_msk_i.astype(np.float32))
        
        if render_scale != 1.0:
            tar_img = cv2.resize(tar_img, None, fx=render_scale, fy=render_scale, interpolation=cv2.INTER_AREA)
            tar_mask = cv2.resize(tar_mask, None, fx=render_scale, fy=render_scale, interpolation=cv2.INTER_NEAREST)
            tar_dpt = cv2.resize(tar_dpt, None, fx=render_scale, fy=render_scale, interpolation=cv2.INTER_NEAREST)
        
        src_views = {'rgb': src_inps, 
                     'extrinsics': src_exts, 
                     'intrinsics': src_ixts}
        tar_views = {'extrinsics': tar_ext, 
                     'intrinsics': tar_ixt, 
                     'rgb': tar_img, 
                     'mask': tar_mask}
        
        near_far = np.array([2.5, 5.5], dtype=np.float32)

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
            img = self.read_image(scene, idx)
            imgs.append(img)
            ixt, ext = self.read_cam(scene, idx)
            ixts.append(ixt)
            exts.append(ext)
        return np.stack(imgs).transpose((0, 3, 1, 2)), np.stack(exts), np.stack(ixts)

    def read_tar(self, scene, view_idx):
        img = self.read_image(scene, view_idx)
        ixt, ext = self.read_cam(scene, view_idx)
        mask = np.ones_like(img[..., 0]).astype(np.uint8)
        return img, mask, ext, ixt

    def read_cam(self, scene, view_idx):
        ext = scene['exts'][view_idx]
        ixt = scene['ixts'][view_idx]
        return ixt, ext

    def read_image(self, scene, view_idx):
        img_path = scene['img_paths'][view_idx]
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.
        img = img[..., :3] * img[..., -1:] + (1 - img[..., -1:])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def __len__(self):
        return len(self.metas)

