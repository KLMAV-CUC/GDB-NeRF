import torch
from utils.optimizer.radam import RAdam


_optimizer_factory = {
    'adam': torch.optim.Adam,
    'adamw': torch.optim.AdamW,
    'radam': RAdam,
    'sgd': torch.optim.SGD
}


def make_optimizer(cfg, net):
    params = []
    lr = cfg.train.lr
    weight_decay = cfg.train.weight_decay
    eps = cfg.train.eps

    for key, value in net.named_parameters():
        if not value.requires_grad:
            continue
        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay, "eps": eps}]

    if 'adam' in cfg.train.optim:
        optimizer = _optimizer_factory[cfg.train.optim](params, lr, weight_decay=weight_decay, eps=eps)
    else:
        optimizer = _optimizer_factory[cfg.train.optim](params, lr, momentum=0.9)

    return optimizer
