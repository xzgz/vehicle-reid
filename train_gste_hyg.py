from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import time
import datetime
import scipy.io
import os.path as osp
import numpy as np

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler

import data_manager
import model_config
import transforms as T
from data_manager.dataset_loader import ClassSampler, ImageDatasetWGL
from model import init_model
from loss.triplet_loss import *
from loss.utils import *
from utils.iotools import save_checkpoint, check_isfile
from utils.logger import Logger
from utils.torchtools import set_bn_to_eval, count_num_param
from optimizers import init_optim
from train_model import train
from test_model import test, test_vehicleid, test_vehicleid_formal


config = model_config.init_dataset_config('veri776')
# config = model_config.init_dataset_config('vehicleid')
config.parameter_init()

data_mean = [0.485, 0.456, 0.406]
data_std = [0.229, 0.224, 0.225]

if config.dataset == 'vehicleid':
    test_model = test_vehicleid
    mean_std = scipy.io.loadmat(osp.join(config.data_root, 'data_mean_std/vehicleid_train_image_mean_std.mat'))
    data_mean = mean_std['mean'][0].astype(np.float64)
    data_std = mean_std['std'][0].astype(np.float64)
else:
    test_model = test
    mean_std = scipy.io.loadmat(osp.join(config.data_root, 'data_mean_std/veri776_train_image_mean_std.mat'))
    data_mean = mean_std['mean'][0].astype(np.float64)
    data_std = mean_std['std'][0].astype(np.float64)

if config.dataset == 'vehicleid' and config.evaluate_cmc:
    test_model = test_vehicleid_formal

def main():
    torch.manual_seed(1)
    os.environ['CUDA_VISIBLE_DEVICES'] = config.gpu_devices
    use_gpu = torch.cuda.is_available()

    sys.stdout = Logger(config.save_dir, config.checkpoint_suffix, config.evaluate)
    print("\n==========\nArgs:")
    config.print_parameter()
    print("==========\n")

    if use_gpu:
        print("Currently using GPU {}".format(config.gpu_devices))
        cudnn.benchmark = True
        torch.cuda.manual_seed_all(1)
    else:
        print("Currently using CPU (GPU is highly recommended)")

    print("Initializing dataset {}".format(config.dataset))
    if config.dataset == 'vehicleid':
        dataset = data_manager.init_imgreid_dataset(name=config.dataset, root=config.data_root, test_size=config.test_size)
    else:
        dataset = data_manager.init_imgreid_dataset(name=config.dataset, root=config.data_root)

    transform_train = T.Compose([
        T.Random2DTranslation(config.height, config.width),
        T.RandomHorizontalFlip(),
        T.Resize((config.height, config.width)),
        T.ToTensor(),
        T.Normalize(mean=data_mean, std=data_std),
    ])

    transform_test = T.Compose([
        T.Resize((config.height, config.width)),
        T.ToTensor(),
        T.Normalize(mean=data_mean, std=data_std),
    ])

    pin_memory = True if use_gpu else False

    # train_batch_sampler = BalancedBatchSampler(dataset.train, n_classes=8, n_samples=8)
    # train_batch_sampler = CCLBatchSampler(dataset.train, n_classes=n_classes, n_samples=n_samples)
    # train_batch_sampler = CCLBatchSamplerV2(dataset.train, n_classes=n_classes, pos_samp_cnt=pos_samp_cnt,
    #                                         neg_samp_cnt=neg_samp_cnt, each_cls_max_cnt=each_cls_max_cnt)
    train_batch_sampler = ClassSampler(dataset.train, sample_cls_cnt=config.sample_cls_cnt, each_cls_cnt=config.each_cls_cnt)

    # trainloader = DataLoader(
    #     ImageDataset(dataset.train, transform=transform_train),
    #     batch_sampler=train_batch_sampler, batch_size=args.train_batch,
    #     shuffle=True, num_workers=args.workers, pin_memory=pin_memory, drop_last=True
    # )

    trainloader = DataLoader(
        ImageDatasetWGL(dataset, data_type='train', transform=transform_train),
        batch_sampler=train_batch_sampler, num_workers=config.workers, pin_memory=pin_memory
    )

    queryloader = DataLoader(
        ImageDatasetWGL(dataset.query, data_type='query', transform=transform_test),
        batch_size=config.test_batch, shuffle=False, num_workers=config.workers,
        pin_memory=pin_memory, drop_last=False,
    )

    galleryloader = DataLoader(
        ImageDatasetWGL(dataset.gallery, data_type='gallery', transform=transform_test),
        batch_size=config.test_batch, shuffle=False, num_workers=config.workers,
        pin_memory=pin_memory, drop_last=False,
    )

    if config.dataset == 'vehicleid':
        train_query_loader = None
        train_gallery_loader = None
        probeloader = DataLoader(
            ImageDatasetWGL(dataset.probe, data_type='probe', transform=transform_test),
            batch_size=config.test_batch, shuffle=False, num_workers=config.workers,
            pin_memory=pin_memory, drop_last=False,
        )
    else:
        train_query_loader = DataLoader(
            ImageDatasetWGL(dataset.train_query, data_type='train_query', transform=transform_test),
            batch_size=config.test_batch, shuffle=False, num_workers=config.workers,
            pin_memory=pin_memory, drop_last=False,
        )

        train_gallery_loader = DataLoader(
            ImageDatasetWGL(dataset.train_gallery, data_type='train_gallery', transform=transform_test),
            batch_size=config.test_batch, shuffle=False, num_workers=config.workers,
            pin_memory=pin_memory, drop_last=False,
        )

    print("Initializing model: {}".format(config.arch))
    model = init_model(name=config.arch, num_classes=dataset.num_train_pids, loss_type=config.loss_type)
    print("Model size: {:.3f} M".format(count_num_param(model)))

    if config.loss_type == 'xent':
        criterion = nn.CrossEntropyLoss()
    elif config.loss_type == 'xent_triplet':
        criterion = XentTripletLoss(margin=config.margin,
                                    triplet_selector=RandomNegativeTripletSelector(margin=config.margin),
                                    each_cls_cnt=config.each_cls_cnt, n_class=config.sample_cls_cnt)
    elif config.loss_type == 'xent_triplet_squa':
        criterion = XentTripletLossSqua(margin=config.margin,
                                        triplet_selector=RandomNegativeTripletSelector(margin=config.margin),
                                        each_cls_cnt=config.each_cls_cnt, n_class=config.sample_cls_cnt)
    elif config.loss_type == 'xent_tripletv2':
        criterion = XentTripletLossV2(margin=config.margin,
                                      triplet_selector=RandomNegativeTripletSelectorV2(margin=config.margin),
                                      each_cls_cnt=config.each_cls_cnt, n_class=config.sample_cls_cnt)
    elif config.loss_type == 'xent_triplet_sqrt':
        criterion = XentTripletLossSqrt(margin=config.margin,
                                        triplet_selector=RandomNegativeTripletSelectorV2(margin=config.margin),
                                        each_cls_cnt=config.each_cls_cnt, n_class=config.sample_cls_cnt)
        # criterion = XentTripletLossV2(margin=0.01, triplet_selector=RandomNegativeTripletSelectorV2(margin=0.01),
        #                               each_cls_cnt=config.each_cls_cnt, n_class=config.sample_cls_cnt)
        # criterion = XentGroupTripletLossV2(margin=0.8, triplet_selector=AllTripletSelector(margin=0.8),
        #                               each_cls_cnt=config.each_cls_cnt, n_class=config.sample_cls_cnt)
    else:
        raise KeyError("Unsupported loss: {}".format(config.loss_type))


    optimizer = init_optim(config.optim, model.parameters(), config.lr, config.weight_decay)
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=config.stepsize, gamma=config.gamma)

    if config.resume is not None:
        if check_isfile(config.resume):
            checkpoint = torch.load(config.resume)
            pretrain_dict = checkpoint['state_dict']
            model_dict = model.state_dict()
            pretrain_dict = {k: v for k, v in pretrain_dict.items() if
                             k in model_dict and model_dict[k].size() == v.size()}
            model_dict.update(pretrain_dict)
            model.load_state_dict(model_dict)
            config.start_epoch = checkpoint['epoch']
            rank1 = checkpoint['rank1']
            if 'mAP' in checkpoint:
                mAP = checkpoint['mAP']
            else:
                mAP = 0
            print("Loaded checkpoint from '{}'".format(config.resume))
            print("- start_epoch: {}\n- rank1: {}\n- mAP: {}".format(config.start_epoch, rank1, mAP))

    if use_gpu:
        model = nn.DataParallel(model).cuda()

    if config.evaluate_cmc:
        if config.dataset == 'vehicleid':
            testloader = probeloader
        else:
            testloader = galleryloader
    else:
        testloader = queryloader
    if config.evaluate:
        print("Evaluate only")
        test_model(model, testloader, galleryloader, train_query_loader, train_gallery_loader, use_gpu,
                   config.test_batch, config.loss_type, config.euclidean_distance_loss, config.start_epoch)
        return

    start_time = time.time()
    train_time = 0
    best_rank1 = -np.inf
    best_map = 0
    best_epoch = 0

    for epoch in range(config.start_epoch, config.max_epoch):
        print("==> Start training")
        start_train_time = time.time()
        scheduler.step()
        print('epoch:', epoch, 'lr:', scheduler.get_lr())
        train(epoch, model, criterion, optimizer, trainloader, config.loss_type, config.print_freq)
        train_time += round(time.time() - start_train_time)

        if epoch >= config.start_eval and config.eval_step > 0 and epoch % config.eval_step == 0 \
           or epoch == config.max_epoch - 1:
            print("==> Test")
            rank1, mAP = test_model(model, testloader, galleryloader, train_query_loader, train_gallery_loader, use_gpu,
                                    config.test_batch, config.loss_type, config.euclidean_distance_loss, epoch + 1)
            is_best = rank1 > best_rank1

            if is_best:
                best_rank1 = rank1
                best_map = mAP
                best_epoch = epoch + 1

            if use_gpu:
                state_dict = model.module.state_dict()
            else:
                state_dict = model.state_dict()

            save_checkpoint({
                'state_dict': state_dict,
                'rank1': rank1,
                'mAP': mAP,
                'epoch': epoch + 1,
            }, is_best, use_gpu_suo=False,
            fpath=osp.join(config.save_dir, 'checkpoint_ep' + str(epoch + 1) + config.checkpoint_suffix + '.pth.tar'))

    print("==> Best Rank-1 {:.2%}, mAP {:.2%}, achieved at epoch {}".format(best_rank1, best_map, best_epoch))
    elapsed = round(time.time() - start_time)
    elapsed = str(datetime.timedelta(seconds=elapsed))
    train_time = str(datetime.timedelta(seconds=train_time))
    print("Finished. Total elapsed time (h:m:s): {}. Training time (h:m:s): {}.".format(elapsed, train_time))


if __name__ == '__main__':
    main()


