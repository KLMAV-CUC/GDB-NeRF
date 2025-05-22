import os
import time
import sys
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict
from termcolor import colored
from configs import cfg


def load_model(net,
               optim,
               scheduler,
               recorder,
               model_dir,
               resume=True,
               epoch=-1):
    if not resume:
        os.system('rm -rf {}'.format(model_dir))

    if not os.path.exists(model_dir):
        return 0

    pths = [
        int(pth.split('.')[0]) for pth in os.listdir(model_dir)
        if pth != 'latest.pth'
    ]
    if len(pths) == 0 and 'latest.pth' not in os.listdir(model_dir):
        return 0
    if epoch == -1:
        if 'latest.pth' in os.listdir(model_dir):
            pth = 'latest'
        else:
            pth = max(pths)
    else:
        pth = epoch
    print('load model: {}'.format(os.path.join(model_dir,
                                               '{}.pth'.format(pth))))
    pretrained_model = torch.load(
        os.path.join(model_dir, '{}.pth'.format(pth)), 'cpu')
    net.load_state_dict(pretrained_model['net'])
    if 'optim' in pretrained_model:
        optim.load_state_dict(pretrained_model['optim'])
        scheduler.load_state_dict(pretrained_model['scheduler'])
        recorder.load_state_dict(pretrained_model['recorder'])
        return pretrained_model['epoch'] + 1
    else:
        return 0


def save_model(net, optim, scheduler, recorder, model_dir, epoch, last=False):
    os.system('mkdir -p {}'.format(model_dir))
    model = {
        'net': net.state_dict(),
        'optim': optim.state_dict(),
        'scheduler': scheduler.state_dict(),
        'recorder': recorder.state_dict(),
        'epoch': epoch
    }
    if last:
        torch.save(model, os.path.join(model_dir, 'latest.pth'))
    else:
        torch.save(model, os.path.join(model_dir, '{}.pth'.format(epoch)))

    # remove previous pretrained model if the number of models is too big
    pths = [
        int(pth.split('.')[0]) for pth in os.listdir(model_dir)
        if pth != 'latest.pth'
    ]
    if len(pths) <= 100:
        return
    os.system('rm {}'.format(
        os.path.join(model_dir, '{}.pth'.format(min(pths)))))


def load_network(net, model_dir, resume=True, epoch=-1, strict=True):
    if not resume:
        return 0

    if not os.path.exists(model_dir):
        print(colored('pretrained model does not exist', 'red'))
        return 0

    if os.path.isdir(model_dir):
        pths = [
            int(pth.split('.')[0]) for pth in os.listdir(model_dir)
            if pth != 'latest.pth'
        ]
        if len(pths) == 0 and 'latest.pth' not in os.listdir(model_dir):
            return 0
        if epoch == -1:
            if 'latest.pth' in os.listdir(model_dir):
                pth = 'latest'
            else:
                pth = max(pths)
        else:
            pth = epoch
        model_path = os.path.join(model_dir, '{}.pth'.format(pth))
    else:
        model_path = model_dir

    print('load model: {}'.format(model_path))
    pretrained_model = torch.load(model_path)
    net.load_state_dict(pretrained_model['net'], strict=strict)
    if 'epoch' in pretrained_model:
        return pretrained_model['epoch'] + 1
    else:
        return 0


def load_pretrain(net, model_dir):
    model_dir = os.path.join(cfg.workspace, 'trained_model', cfg.task, model_dir)
    if not os.path.exists(model_dir):
        return 1

    pths = [int(pth.split('.')[0]) for pth in os.listdir(model_dir) if pth != 'latest.pth']
    if len(pths) == 0 and 'latest.pth' not in os.listdir(model_dir):
        return 1

    if 'latest.pth' in os.listdir(model_dir):
        pth = 'latest'
    else:
        pth = max(pths)

    print('Load pretrain model: {}'.format(os.path.join(model_dir, '{}.pth'.format(pth))))
    pretrained_model = torch.load(os.path.join(model_dir, '{}.pth'.format(pth)), 'cpu')
    net.load_state_dict(pretrained_model['net'])
    return 0


def save_pretrain(net, task, model_dir):
    model_dir = os.path.join('data/trained_model', task, model_dir)
    os.system('mkdir -p ' +  model_dir)
    model = {'net': net.state_dict()}
    torch.save(model, os.path.join(model_dir, 'latest.pth'))


