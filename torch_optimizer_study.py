from __future__ import print_function
from __future__ import division

import os
import numpy as np

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.optim import lr_scheduler

from model import init_model
from utils.torchtools import count_num_param
from optimizers import init_optim

gpu_devices = '0'
use_cpu = False
arch = 'resnet50'
loss_type = 'xent'
lr = 2e-2
gamma = 0.1
weight_decay = 2e-4
start_epoch = 0
max_epoch = 12
stepsize = [8, 10]
optim = 'sgd'


def main():
    torch.manual_seed(1)
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_devices
    use_gpu = torch.cuda.is_available()
    if use_cpu: use_gpu = False

    if use_gpu:
        print("Currently using GPU {}".format(gpu_devices))
        cudnn.benchmark = True
        torch.cuda.manual_seed_all(1)
    else:
        print("Currently using CPU (GPU is highly recommended)")

    print("Initializing model: {}".format(arch))
    model = init_model(name=arch, num_classes=576, loss_type=loss_type)
    print("Model size: {:.3f} M".format(count_num_param(model)))
    if use_gpu:
        model = nn.DataParallel(model).cuda()

    optimizer = init_optim(optim, model.parameters(), lr, weight_decay)
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=stepsize, gamma=gamma)

    model.train()
    cnt = 0
    for epoch in range(start_epoch, max_epoch, 2):
        for step in range(2):
            x = torch.randn(1, 3, 200, 200)
            y = torch.randint(low=0, high=576, size=(1,), dtype=torch.int64)
            if use_gpu:
                x = x.cuda()
                y = y.cuda()
            scheduler.step()
            cnt += 1
            print(cnt, scheduler.get_lr())
            output = model(x)
            # loss = nn.CrossEntropyLoss()(output[0], y)
            loss = torch.tensor(0.0, dtype=torch.float32).cuda()
            # loss = torch.tensor(0.0, dtype=torch.float32)
            loss.requires_grad = True
            optimizer.zero_grad()
            loss.backward()
            print(loss)
            print(loss._grad)
            optimizer.step()
    print('Done.')
main()

