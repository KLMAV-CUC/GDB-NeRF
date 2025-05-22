import random
import tqdm
import torch
import time
import numpy as np
from configs import cfg, args
from datasets import make_data_loader
from networks import make_network
from utils.data_utils import to_cuda
from evaluators import make_evaluator
from utils import net_utils


if cfg.fix_random:
    random.seed(0)
    np.random.seed(0)


def run_dataset():
    cfg.train.num_workers = 0
    data_loader = make_data_loader(cfg, is_train=False)
    for batch in tqdm.tqdm(data_loader):
        pass


def run_network():
    network = make_network(cfg).cuda()
    net_utils.load_network(network, cfg.trained_model_dir, epoch=cfg.test.epoch)
    network.eval()

    data_loader = make_data_loader(cfg, is_train=False)
    total_time = 0
    for batch in tqdm.tqdm(data_loader):
        batch = to_cuda(batch)
        with torch.no_grad():
            torch.cuda.synchronize()
            start = time.time()
            network(batch)
            torch.cuda.synchronize()
            total_time += time.time() - start
    print(total_time / len(data_loader))

def run_evaluate():
    network = make_network(cfg).cuda()
    net_utils.load_network(network,
                           cfg.trained_model_dir,
                           resume=cfg.resume,
                           epoch=cfg.test.epoch)
    network.eval()

    data_loader = make_data_loader(cfg, is_train=False)
    evaluator = make_evaluator(cfg)
    net_time = []
    for batch in tqdm.tqdm(data_loader):
        batch = to_cuda(batch)
        # for k in batch:
        #     if k != 'meta':
        #         batch[k] = batch[k].cuda()
        with torch.no_grad():
            torch.cuda.synchronize()
            start_time = time.time()
            output, _, _ = network(batch)
            torch.cuda.synchronize()
            end_time = time.time()
        net_time.append(end_time - start_time)
        evaluator.evaluate(output, batch)
    evaluator.summarize()
    if len(net_time) > 1:
        # print('net_time: ', np.mean(net_time[1:]))
        print('FPS: ', 1./np.mean(net_time[1:]))
    else:
        # print('net_time: ', np.mean(net_time))
        print('FPS: ', 1./np.mean(net_time))


if __name__ == '__main__':
    globals()['run_' + args.type]()
