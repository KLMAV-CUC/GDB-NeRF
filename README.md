# Depth-Guided Bundle Sampling for Efficient Generalizable Neural Radiance Field Reconstruction

Official PyTorch implementation of the paper:

> [Depth-Guided Bundle Sampling for Efficient Generalizable Neural Radiance Field Reconstruction](http://arxiv.org/abs/2505.19793) \
> Li Fang, Hao Zhu, Longlong Chen, Fei Hu, Long Ye, and Zhan Ma \
> CVPR 2025

GDB-NeRF refers to ENeRF+Ours in the paper. The proposed bundle sampling strategy can be used in other NeRFs.

## Installation
Please install the following dependencies first
- [PyTorch (2.5.1 + cu11.8)](https://pytorch.org/get-started/previous-versions/)
- [nvdiffrast](https://nvlabs.github.io/nvdiffrast/)

And then install the other dependencies using
```
pip install -r requirements.txt
```

### Set up datasets

#### 0. Set up workspace
The workspace is the disk directory that stores datasets, training logs, checkpoints and results. Please ensure it has enough space. 
```
export workspace=$PATH_TO_YOUR_WORKSPACE
```
   
#### 1. Pre-trained model

Download the pretrained model from [dtu_pretrain](https://drive.google.com/drive/folders/1mPbNpBnIYIbC-5wlbGSkx8kXefZMZ4Re?usp=drive_link) (Pretrained on DTU dataset.)

Put it into $workspace/trained_model/gdb_nerf/dtu_pretrain/latest.pth

#### 2. DTU, Real Forward-facing and NeRF Synthetic
Please refer to [ENeRF](https://github.com/zju3dv/enerf) for the dataset preparation.

## Training and fine-tuning

### Training
Use the following command to train a generalizable model on DTU.
```
python train_net.py --cfg_file configs/dtu_pretrain.yaml
```

### Fine-tuning

```
mkdir dtu_ft_scan114
cp dtu_pretrain/233.pth dtu_ft_scan114
python train_net.py --cfg_file configs/dtu/scan114.yaml
```

## Evaluation

### Evaluate the pretrained model on DTU

Use the following command to evaluate the pretrained model on DTU.
```
python run.py --type evaluate --cfg_file configs/dtu_eval.yaml test.eval_depth True
```

**Add the "save_result True" parameter at the end of the command to save the rendering result.**

### Evaluate the pretrained model on LLFF and NeRF datasets

```
python run.py --type evaluate --cfg_file configs/nerf_eval.yaml
```

```
python run.py --type evaluate --cfg_file configs/llff_eval.yaml
```

## Citation

If you find this code useful for your research, please use the following BibTeX entry.

```
@inproceedings{fang2025depth,
  title={Depth-Guided Bundle Sampling for Efficient Generalizable Neural Radiance Field Reconstruction},
  author={Fang, Li and Zhu, Hao and Chen, Longlong and Hu, Fei and Ye, Long and Ma, Zhan},
  booktitle={CVPR},
  year={2025}
}
```

## Acknowledgements
Thanks to Haotong Lin and Sida Peng for opening source of their excellent works [ENeRF](https://github.com/zju3dv/enerf/).
