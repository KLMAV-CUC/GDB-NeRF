import os
import random
import cv2
import torch
import numpy as np
from operator import itemgetter
from utils import data_utils


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
        self.num_depth = cfg.nerf.global_num_depth
        # self.interval_scale = 1.06 / (float(self.num_depth)/192.0)
        self.interval_scale = 1. / (float(self.num_depth)/192.0)
        self.build_metas(kwargs['ann_file'])

    def build_metas(self, ann_file):
        scenes = [line.strip() for line in open(ann_file).readlines()]
        dtu_pairs = torch.load('data/mvsnerf/pairs.th')

        self.scene_infos = {}
        self.metas = []
        if len(self.scenes) != 0:
            scenes = self.scenes

        for scene in scenes:
            scene_info = {'ixts': [], 'exts': [], 'dpt_paths': [], 'img_paths': [], 'near_far': []}
            for i in range(49):
                cam_path = os.path.join(self.data_root, 'Cameras/train/{:08d}_cam.txt'.format(i))
                ixt, ext, depth_min, depth_interval = data_utils.read_cam_file(cam_path)
                ext[:3, 3] = ext[:3, 3]
                ixt[:2] = ixt[:2] * 4
                dpt_path = os.path.join(self.data_root, 'Depths/{}/depth_map_{:04d}.pfm'.format(scene, i))
                img_path = os.path.join(self.data_root, 'Rectified/{}_train/rect_{:03d}_3_r5000.png'.format(scene, i+1))

                # Get sampling interval
                depth_max = depth_min + depth_interval * self.interval_scale * self.num_depth

                scene_info['ixts'].append(ixt.astype(np.float32))
                scene_info['exts'].append(ext.astype(np.float32))
                scene_info['dpt_paths'].append(dpt_path)
                scene_info['img_paths'].append(img_path)
                scene_info['near_far'].append(np.array([depth_min, depth_max], dtype=np.float32))

            if self.split == 'train' and len(self.scenes) != 1:
                train_ids = np.arange(49).tolist()
                test_ids = np.arange(49).tolist()
            elif self.split == 'train' and len(self.scenes) == 1:
                train_ids = dtu_pairs['dtu_train']
                test_ids = dtu_pairs['dtu_train']
            else:
                train_ids = dtu_pairs['dtu_train']
                test_ids = dtu_pairs['dtu_val']
            scene_info.update({'train_ids': train_ids, 'test_ids': test_ids})
            self.scene_infos[scene] = scene_info

            cam_points = np.array([np.linalg.inv(scene_info['exts'][i])[:3, 3] for i in train_ids])
            for tar_view in test_ids:
                cam_point = np.linalg.inv(scene_info['exts'][tar_view])[:3, 3]
                distance = np.linalg.norm(cam_points - cam_point[None], axis=-1)
                argsorts = distance.argsort()
                argsorts = argsorts[1:] if tar_view in train_ids else argsorts
                input_views_num = max(self.cfg.train.sampler_meta.input_views_num) if self.split == 'train' else self.cfg.test.sampler_meta.input_views_num[0]
                src_views = [train_ids[i] for i in argsorts[:input_views_num]]
                self.metas += [(scene, tar_view, src_views)]

    def __getitem__(self, index_meta):
        index, input_views_num, render_scale = index_meta
        scene, tar_view, src_views = self.metas[index]
        if self.split == 'train':
            if random.random() < 0.1:
                src_views = src_views + [tar_view]
            src_views = random.sample(src_views[:input_views_num+1], input_views_num)
        scene_info = self.scene_infos[scene]

        tar_img = cv2.cvtColor(cv2.imread(scene_info['img_paths'][tar_view], cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.
        H, W = tar_img.shape[:2]
        meta = {'scene': scene, 'tar_view': tar_view, 'frame_id': 0, 'h': H, 'w': W}
        tar_ext, tar_ixt = scene_info['exts'][tar_view], scene_info['ixts'][tar_view]
        
        tar_dpt = data_utils.read_pfm(scene_info['dpt_paths'][tar_view])[0].astype(np.float32)
        tar_dpt = cv2.resize(tar_dpt, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_NEAREST)
        tar_dpt = tar_dpt[44:556, 80:720]
        tar_mask = (tar_dpt > 0.).astype(np.uint8)

        if render_scale != 1.0:
            tar_img = cv2.resize(tar_img, None, fx=render_scale, fy=render_scale, interpolation=cv2.INTER_AREA)
            tar_mask = cv2.resize(tar_mask, None, fx=render_scale, fy=render_scale, interpolation=cv2.INTER_NEAREST)
            tar_dpt = cv2.resize(tar_dpt, None, fx=render_scale, fy=render_scale, interpolation=cv2.INTER_NEAREST)
        
        src_inps, src_exts, src_ixts = self.read_src(scene_info, src_views)

        tar_gt_ms = {'rgb': [], 
                     'mask': [], 
                     'depth': []}
        
        # Multi-scale GT for depth estimation
        for s in self.cfg.mvs.vol_scales:
            tar_img_i = cv2.resize(tar_img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
            tar_msk_i = cv2.resize(tar_mask, None, fx=s, fy=s, interpolation=cv2.INTER_NEAREST)
            tar_dpt_i = cv2.resize(tar_dpt, None, fx=s, fy=s, interpolation=cv2.INTER_NEAREST)
            tar_gt_ms['rgb'].append(tar_img_i)
            tar_gt_ms['mask'].append(tar_msk_i)
            tar_gt_ms['depth'].append(tar_dpt_i)
        
        src_views = {'rgb': src_inps, 
                     'extrinsics': src_exts, 
                     'intrinsics': src_ixts}
        tar_views = {'extrinsics': tar_ext, 
                     'intrinsics': tar_ixt, 
                     'rgb': tar_img, 
                     'mask': tar_mask, 
                     'depth': tar_dpt}
        
        near_far = scene_info['near_far'][tar_view]

        return {'src_views': src_views, 
                'tar_views': tar_views, 
                'near_far': near_far, 
                'tar_gt_ms': tar_gt_ms, 
                'render_scale': render_scale, 
                'meta': meta}

    def read_src(self, scene_info, src_views):
        inps, exts, ixts = [], [], []
        for src_view in src_views:
            inps.append(cv2.cvtColor(cv2.imread(scene_info['img_paths'][src_view], cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.)
            exts.append(scene_info['exts'][src_view])
            ixts.append(scene_info['ixts'][src_view])
        return np.stack(inps).transpose((0, 3, 1, 2)), np.stack(exts), np.stack(ixts)

    def __len__(self):
        return len(self.metas)

