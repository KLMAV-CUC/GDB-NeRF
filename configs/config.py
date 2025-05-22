import os
import yaml
import argparse
import numpy as np
from ast import literal_eval
from types import SimpleNamespace


def _decode_cfg_value(v):
    """Decodes a raw config value (e.g., from a yaml config files or command
    line argument) into a Python object.
    """
    # All remaining processing is only applied to strings
    if not isinstance(v, str):
        return v
    # Try to interpret `v` as a:
    #   string, number, tuple, list, dict, boolean, or None
    try:
        v = literal_eval(v)
    # The following two excepts allow v to pass through when it represents a
    # string.
    #
    # Longer explanation:
    # The type of v is always a string (before calling literal_eval), but
    # sometimes it *represents* a string and other times a data structure, like
    # a list. In the case that v represents a string, what we got back from the
    # yaml parser is 'foo' *without quotes* (so, not '"foo"'). literal_eval is
    # ok with '"foo"', but will raise a ValueError if given 'foo'. In other
    # cases, like paths (v = 'foo/bar' and not v = '"foo/bar"'), literal_eval
    # will raise a SyntaxError.
    except (ValueError, SyntaxError):
        pass
    return v


def _dotdictify(my_dict):
    new_dict = SimpleNamespace(**my_dict)
    for k, v in my_dict.items():
        if isinstance(v, dict):
            setattr(new_dict, k, _dotdictify(v))
    return new_dict


def _merge_dicts(dict1, dict2):
    """Recursively merges dict2 into dict1, overwriting values in dict1."""
    for key, value in dict2.items():
        if isinstance(value, dict) and key in dict1 and isinstance(dict1[key], dict):
            _merge_dicts(dict1[key], value)
        else:
            dict1[key] = value
    return dict1


def make_cfg(cfg, args):
    if args.cfg_file:
        with open(args.cfg_file, 'r', encoding='utf-8') as f:
            yaml_cfg = yaml.load(f, Loader=yaml.FullLoader)

    # Merge default and custom config
    if 'parent_cfg' in yaml_cfg:
        with open(yaml_cfg['parent_cfg'], encoding='utf-8') as f:
            parent_cfg = yaml.load(f, Loader=yaml.FullLoader)
            cfg = _merge_dicts(cfg, parent_cfg)
    cfg = _merge_dicts(cfg, yaml_cfg)

    # Overwrite other operations
    cfg_list = args.opts
    assert len(cfg_list) % 2 == 0, "Override list has odd length: {}; it must be a list of pairs".format(cfg_list)
    for i in range(0, len(cfg_list), 2):
        keys = cfg_list[i].split('.')
        value = _decode_cfg_value(cfg_list[i + 1])
        
        sub_dict = cfg
        for key in keys[:-1]:
            if key not in sub_dict:
                sub_dict[key] = {}
            sub_dict = sub_dict[key]
        sub_dict[keys[-1]] = value

    if len(cfg['task']) == 0:
        raise ValueError('Task must be specified')

    # assign the gpus
    if -1 not in cfg['gpus']:
        os.environ['CUDA_VISIBLE_DEVICES'] = ', '.join([str(gpu) for gpu in cfg['gpus']])

    if 'bbox' in cfg:
        bbox = np.array(cfg['bbox']).reshape((2, 3))
        center, half_size = np.mean(bbox, axis=0), (bbox[1]-bbox[0]).max().item() / 2.
        bbox = np.stack([center-half_size, center+half_size])
        cfg['bbox'] = bbox.reshape(6).tolist()

    if len(cfg['exp_name_tag']) != 0:
        cfg['exp_name'] +=  ('_' + cfg['exp_name_tag'])
    cfg['exp_name'] = cfg['exp_name'].replace('gitbranch', os.popen('git describe --all').readline().strip()[6:])
    cfg['exp_name'] = cfg['exp_name'].replace('gitcommit', os.popen('git describe --tags --always').readline().strip())
    print('EXP NAME: ', cfg['exp_name'])
    cfg['trained_model_dir'] = os.path.join(cfg['trained_model_dir'], cfg['task'], cfg['exp_name'])
    cfg['record_dir'] = os.path.join(cfg['record_dir'], cfg['task'], cfg['exp_name'])
    cfg['result_dir'] = os.path.join(cfg['result_dir'], cfg['task'], cfg['exp_name'], cfg['save_tag'])
    
    modules = [key for key in cfg if '_module' in key]
    for module in modules:
        cfg[module.replace('_module', '_path')] = cfg[module].replace('.', '/') + '.py'
    
    return _dotdictify(cfg)


parser = argparse.ArgumentParser()

parser.add_argument("--cfg_file", default="configs/default.yaml", type=str)
parser.add_argument('--test', action='store_true', dest='test', default=False)
parser.add_argument("--type", type=str, default="")
parser.add_argument('--det', type=str, default='')
parser.add_argument('--local_rank', type=int, default=0)
parser.add_argument("opts", default=None, nargs=argparse.REMAINDER)

args = parser.parse_args()

# Default config
cfg = {
    'workspace': os.environ['workspace'],
    'save_tag': 'default',
    'exp_name_tag': '',  # experiment name
    'trained_model_dir': os.path.join(os.environ['workspace'], 'trained_model'),  # Trained model
    'record_dir': os.path.join(os.environ['workspace'], 'record'),  # Recorder
    'result_dir': os.path.join(os.environ['workspace'], 'result'),  # Result
    'local_rank': args.local_rank,
    # visualization
    'write_video': False,
    'fps': 24,
    # network
    'distributed': False,
    # task
    'task': 'hello',
    # gpus
    'gpus': [0, 1, 2, 3],
    # if load the pretrained network
    'resume': True,
    # epoch
    'ep_iter': -1,
    'save_ep': 1,
    'save_latest_ep': 1,
    'eval_ep': 1,
    'log_interval': 20,
    # dataset
    'sample_on_mask': False,
    # train
    'train': {
        'pretrain': '',
        'epoch': 10000,
        'num_workers': 8,
        'collator': 'default',
        'batch_sampler': 'default',
        'shuffle': True,
        'eps': 1.e-8,
        'sampler_meta': {
            'input_views_num': [],
            'input_views_prob': []
        },
        # use adam as default
        'optim': 'adam',
        'lr': 5.e-4,
        'weight_decay': 0.0,
        'scheduler': {
            'type': 'multi_step',
            'milestones': [80, 120, 200, 240],
            'gamma': 0.5
        },
        'batch_size': 4
    },
    # test
    'test': {
        'batch_size': 1,
        'collator': 'default',
        'epoch': -1,
        'batch_sampler': 'default',
        'sampler_meta': {
            'input_views_num': [],
            'input_views_prob': []
        }
    },
    # evaluation
    'skip_eval': False,
    'fix_random': False
}
print('Workspace: ', cfg['workspace'])

if len(args.type) > 0:
    cfg.update({'task': 'run'})

cfg = make_cfg(cfg, args)

print(cfg)
