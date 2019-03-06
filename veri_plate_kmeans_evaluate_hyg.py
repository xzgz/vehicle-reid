from __future__ import print_function
from __future__ import division

import os
import re
import sys
import time
import datetime
import argparse
import scipy.io
import os.path as osp
import numpy as np

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from torchvision.transforms import functional

from torchreid import data_manager
from torchreid.dataset_loader import *
from torchreid import transforms as T
from torchreid import models
from torchreid.losses import *
from torchreid.utils.iotools import save_checkpoint, check_isfile
from torchreid.utils.avgmeter import AverageMeter
from torchreid.utils.logger import Logger
from torchreid.utils.torchtools import set_bn_to_eval, count_num_param
from torchreid.utils.reidtools import visualize_ranked_results
from torchreid.eval_metrics import evaluate
from torchreid.optimizers import init_optim
from torchreid.utils.tsne import tsne_show
from torchreid.utils.kmeans import KMeans


parser = argparse.ArgumentParser(
    description='Train image model with cross entropy loss')
# Datasets
parser.add_argument('--root', type=str, default='data',
                    help="root path to data directory")
parser.add_argument('-d', '--dataset', type=str, default='market1501',
                    choices=data_manager.get_names())
parser.add_argument('-j', '--workers', default=4, type=int,
                    help="number of data loading workers (default: 4)")
parser.add_argument('--height', type=int, default=256,
                    help="height of an image (default: 256)")
parser.add_argument('--width', type=int, default=128,
                    help="width of an image (default: 128)")
parser.add_argument('--split-id', type=int, default=0,
                    help="split index")
# CUHK03-specific setting
parser.add_argument('--cuhk03-labeled', action='store_true',
                    help="whether to use labeled images, if false, detected images are used (default: False)")
parser.add_argument('--cuhk03-classic-split', action='store_true',
                    help="whether to use classic split by Li et al. CVPR'14 (default: False)")
parser.add_argument('--use-metric-cuhk03', action='store_true',
                    help="whether to use cuhk03-metric (default: False)")
# Optimization options
parser.add_argument('--optim', type=str, default='adam',
                    help="optimization algorithm (see optimizers.py)")
parser.add_argument('--max-epoch', default=60, type=int,
                    help="maximum epochs to run")
parser.add_argument('--start-epoch', default=0, type=int,
                    help="manual epoch number (useful on restarts)")
parser.add_argument('--train-batch', default=32, type=int,
                    help="train batch size")
parser.add_argument('--test-batch', default=100, type=int,
                    help="test batch size")
parser.add_argument('--lr', '--learning-rate', default=0.0003, type=float,
                    help="initial learning rate")
parser.add_argument('--stepsize', default=[20, 40], nargs='+', type=int,
                    help="stepsize to decay learning rate")
parser.add_argument('--gamma', default=0.1, type=float,
                    help="learning rate decay")
parser.add_argument('--weight-decay', default=5e-04, type=float,
                    help="weight decay (default: 5e-04)")
parser.add_argument('--fixbase-epoch', default=0, type=int,
                    help="epochs to fix base network (only train classifier, default: 0)")
parser.add_argument('--fixbase-lr', default=0.0003, type=float,
                    help="learning rate (when base network is frozen)")
parser.add_argument('--freeze-bn', action='store_true',
                    help="freeze running statistics in BatchNorm layers during training (default: False)")
parser.add_argument('--label-smooth', action='store_true',
                    help="use label smoothing regularizer in cross entropy loss")
# Architecture
parser.add_argument('-a', '--arch', type=str, default='resnet50',
                    choices=models.get_names())
parser.add_argument('--loss-type', type=str, default='xent')
# Miscs
parser.add_argument('--print-freq', type=int, default=10,
                    help="print frequency")
parser.add_argument('--seed', type=int, default=1,
                    help="manual seed")
parser.add_argument('--resume', type=str, default='', metavar='PATH')
parser.add_argument('--load-weights', type=str, default='',
                    help="load pretrained weights but ignores layers that don't match in size")
parser.add_argument('--evaluate', action='store_true',
                    help="evaluation only")
parser.add_argument('--eval-step', type=int, default=-1,
                    help="run evaluation for every N epochs (set to -1 to test after training)")
parser.add_argument('--start-eval', type=int, default=0,
                    help="start to evaluate after specific epoch")
parser.add_argument('--save-dir', type=str, default='log')
parser.add_argument('--use-cpu', action='store_true',
                    help="use cpu")
parser.add_argument('--gpu-devices', default='0', type=str,
                    help='gpu device ids for CUDA_VISIBLE_DEVICES')
parser.add_argument('--vis-ranked-res', action='store_true',
                    help="visualize ranked results, only available in evaluation mode (default: False)")

args = parser.parse_args()
# use_gpu_suo = False
use_gpu_suo = True

if use_gpu_suo:
    args.root = '/home/weiying1/hyg/pytorch-workspace/pytorch-study/data'
    args.save_dir = '/home/weiying1/hyg/pytorch-workspace/pytorch-study/log-reid/mgn-xtlrnsv2-c8m8-lr1ef2-sec-plt'

    args.resm_plt = '/home/weiying1/hyg/pytorch-workspace/pytorch-study/log-reid/hacnn-xent-bh5-lr1ef2/best_model.pth.tar'

    args.resm_vecl = '/home/weiying1/hyg/pytorch-workspace/pytorch-study/log-reid/' \
                     'mgn-xtlrnsv2-c8m8-lr1ef2-norm-last-m0.8-thi/checkpoint_ep10.pth.tar'
else:
    args.root = '/home/gysj/pytorch-workspace/pytorch-study/data'
    args.save_dir = '/media/sda1/sleep-data/gysj/log-reid/train-hyg/mgn-xtlrnsv2-c8m8-lr1ef2-sec-plt'

    args.resm_plt = '/media/sda1/sleep-data/gysj/log-reid/train-cheney/hacnn-xent-bh5-lr1ef2/best_model.pth.tar'

    # args.resm_vecl = '/media/sda1/sleep-data/gysj/log-reid/train-hyg/mgn-xtilrns-c8m8-lr1ef2/checkpoint_ep20.pth.tar'
    # args.resm_vecl = '/media/sda1/sleep-data/gysj/log-reid/train-hyg/mgn-xtlrnsv2-c8m8-lr1ef2-sec/checkpoint_ep20.pth.tar'
    args.resm_vecl = '/media/sda1/sleep-data/gysj/log-reid/train-hyg/mgn-xtlrnsv2-c8m8-lr1ef2-norm-last-m0.8-sec/' \
                     'checkpoint_ep13_sf6lr1ef3.pth.tar'


args.arch_plt = 'hacnn'
args.arch_vecl = 'mgn'
args.dataset_plt = 'veri776plt'
args.dataset_vecl = 'veri776wgl'
args.gpu_devices = '1'
args.workers = 0

# args.optim = 'adam'
args.optim = 'sgd'

# args.loss_type = 'angle'
args.loss_type = 'xent'
# args.loss_type = 'triplet'
# args.loss_type = 'tripletv2'
# args.loss_type = 'xent_htri'
euclidean_distance_loss = ['angle', 'xent', 'triplet', 'xent_htri', 'tripletv2']

args.lr = 1e-2

args.eval_step = 2

args.fixbase_lr = 5e-4
args.gamma = 0.1

# args.weight_decay = 0
# args.weight_decay = 5e-4
args.weight_decay = 2e-4

args.start_epoch = 0
# args.max_epoch = 56
# args.max_epoch = 60
args.max_epoch = 200

# args.stepsize = [12, 20]
# args.stepsize = [15, 25]
# args.stepsize = [100]

args.test_batch = 100

# args.height_plt = 20
# args.width_plt = 60
args.height_plt = 160
args.width_plt = 64

args.height_vecl = 256
args.width_vecl = 256

num_clusters = 4
# num_clusters = 5
# top_rerk_num = 30
top_rerk_num = 10

# use_plt = False
use_plt = True
only_use_plt = False
# only_use_plt = True
# vecl_plt_kmeans = True
vecl_plt_kmeans = False

# n_classes = 8
# pos_samp_cnt = 10
# neg_samp_cnt = 10
# each_cls_max_cnt = 18

checkpoint_suffix = ''
# checkpoint_suffix = '_sf30lr1ef3'

args.evaluate = True
# args.use_metric_cuhk03 = True
# args.label_smooth = True

# k_plt = 0
# k_plt = 1
k_plt = 2
# k_plt = 3
# k_plt = 4
if only_use_plt:
    print('Plate only.')
else:
    if use_plt and vecl_plt_kmeans:
        print('k_plt = {}, with kmeans.'.format(k_plt))
    elif use_plt and not vecl_plt_kmeans:
        print('k_plt = {}.'.format(k_plt))
    else:
        print('Use vehicle model only.')


class LoggerInfo():
    def __init__(self):
        self.checkpoint_suffix = checkpoint_suffix
        self.start_epoch, self.train_step = self.get_start_epoch()

        current_log_file = osp.join(args.save_dir,
                                    'log_train' + self.checkpoint_suffix + '.txt')
        previous_log_file = 'log_train'
        if self.train_step >= 1:
            if self.train_step == 1:
                previous_log_file = osp.join(args.save_dir, 'log_train.txt')
            else:
                suffix_list = self.checkpoint_suffix.split('_')
                for i in range(1, len(suffix_list) - 1):
                    previous_log_file += '_'
                    previous_log_file += suffix_list[i]
                previous_log_file += '.txt'
            previous_log_file = osp.join(args.save_dir, previous_log_file)
            previous_log = self.get_previous_log(previous_log_file)
            self.init_current_log_file(current_log_file, previous_log)

        if args.evaluate:
            self.fpath = osp.join(args.save_dir, 'log_test.txt')
        else:
            self.fpath = current_log_file

    def get_start_epoch(self):
        pattern_start_epoch = re.compile('(_sf(\d+))+')
        match = pattern_start_epoch.search(self.checkpoint_suffix)
        if match is not None:
            match_group = match.groups()
            return int(match_group[-1]) + 1, len(match_group)
        return 0, 0

    def get_previous_log(self, previous_log_file):
        pattern_end_line = re.compile('Epoch:\s\[(\d+)')
        f = open(previous_log_file)
        lines = f.readlines()
        f.close()
        previous_log = []
        for i, line in enumerate(lines):
            match = pattern_end_line.search(line.strip())
            if match is not None:
                match_group = match.groups()
                epoch = int(match_group[0])
                if epoch == self.start_epoch:
                    return previous_log
            previous_log.append(line)
        return previous_log

    def init_current_log_file(self, current_log_file, previous_log):
        f = open(current_log_file, 'w')
        for line in previous_log:
            f.write(line)
        f.write('\n\n\n')
        f.close()


def main():
    torch.manual_seed(args.seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_devices
    use_gpu = torch.cuda.is_available()
    if args.use_cpu: use_gpu = False

    logger_info = LoggerInfo()
    sys.stdout = Logger(logger_info)
    print("==========\nArgs:{}\n==========".format(args))

    if use_gpu:
        print("Currently using GPU {}".format(args.gpu_devices))
        cudnn.benchmark = True
        torch.cuda.manual_seed_all(args.seed)
    else:
        print("Currently using CPU (GPU is highly recommended)")

    print("\nInitializing dataset {}".format(args.dataset_plt))
    dataset_plt = data_manager.init_imgreid_dataset(root=args.root, name=args.dataset_plt)
    print("\nInitializing dataset {}".format(args.dataset_vecl))
    dataset_vecl = data_manager.init_imgreid_dataset(root=args.root, name=args.dataset_vecl)

    transform_test_plt = T.Compose([
        T.Resize((args.height_plt, args.width_plt)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    # transform_flip_test_plt = T.Compose([
    #     T.Resize((args.height_plt, args.width_plt)),
    #     functional.hflip,
    #     T.ToTensor(),
    #     T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    # ])
    transform_test_vecl = T.Compose([
        T.Resize((args.height_vecl, args.width_vecl)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    # transform_flip_test_vecl = T.Compose([
    #     T.Resize((args.height_vecl, args.width_vecl)),
    #     functional.hflip,
    #     T.ToTensor(),
    #     T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    # ])

    pin_memory = True if use_gpu else False

    queryloader_plt = DataLoader(
        ImageDatasetV2(dataset_plt.query, transform=transform_test_plt),
        batch_size=args.test_batch, shuffle=False, num_workers=args.workers,
        pin_memory=pin_memory, drop_last=False,
    )
    # queryloader_flip_plt = DataLoader(
    #     ImageDatasetV2(dataset_plt.query, transform=transform_flip_test_plt),
    #     batch_size=args.test_batch, shuffle=False, num_workers=args.workers,
    #     pin_memory=pin_memory, drop_last=False,
    # )
    # queryloader_plt = [queryloader_plt, queryloader_flip_plt]
    queryloader_plt = [queryloader_plt]
    galleryloader_plt = DataLoader(
        ImageDatasetV2(dataset_plt.gallery, transform=transform_test_plt),
        batch_size=args.test_batch, shuffle=False, num_workers=args.workers,
        pin_memory=pin_memory, drop_last=False,
    )
    # galleryloader_flip_plt = DataLoader(
    #     ImageDatasetV2(dataset_plt.gallery, transform=transform_flip_test_plt),
    #     batch_size=args.test_batch, shuffle=False, num_workers=args.workers,
    #     pin_memory=pin_memory, drop_last=False,
    # )
    # galleryloader_plt = [galleryloader_plt, galleryloader_flip_plt]
    galleryloader_plt = [galleryloader_plt]

    queryloader_vecl = DataLoader(
        ImageDatasetWGL(dataset_vecl.query, transform=transform_test_vecl, with_image_name=True),
        batch_size=args.test_batch, shuffle=False, num_workers=args.workers,
        pin_memory=pin_memory, drop_last=False,
    )
    # queryloader_flip_vecl = DataLoader(
    #     ImageDatasetV2(dataset_vecl.query, transform=transform_flip_test_vecl),
    #     batch_size=args.test_batch, shuffle=False, num_workers=args.workers,
    #     pin_memory=pin_memory, drop_last=False,
    # )
    # queryloader_vecl = [queryloader_vecl, queryloader_flip_vecl]
    queryloader_vecl = [queryloader_vecl]
    galleryloader_vecl = DataLoader(
        ImageDatasetWGL(dataset_vecl.gallery, transform=transform_test_vecl, with_image_name=True),
        batch_size=args.test_batch, shuffle=False, num_workers=args.workers,
        pin_memory=pin_memory, drop_last=False,
    )
    # galleryloader_flip_vecl = DataLoader(
    #     ImageDatasetV2(dataset_vecl.gallery, transform=transform_flip_test_vecl),
    #     batch_size=args.test_batch, shuffle=False, num_workers=args.workers,
    #     pin_memory=pin_memory, drop_last=False,
    # )
    # galleryloader_vecl = [galleryloader_vecl, galleryloader_flip_vecl]
    galleryloader_vecl = [galleryloader_vecl]

    print("\nInitializing model: {}".format(args.arch))
    model_plt = models.init_model(name=args.arch_plt, num_classes=dataset_plt.num_train_pids, loss_type=args.loss_type)
    model_vecl = models.init_model(name=args.arch_vecl, num_classes=dataset_vecl.num_train_pids, loss_type=args.loss_type)
    print("Plate model size: {:.3f} M".format(count_num_param(model_plt)))
    print("Vehicle model size: {:.3f} M".format(count_num_param(model_vecl)))

    if args.loss_type == 'xent':
        criterion = nn.CrossEntropyLoss()
    else:
        raise KeyError("Unsupported loss: {}".format(args.loss_type))

    if args.resm_plt and args.resm_vecl:
        if check_isfile(args.resm_plt) and check_isfile(args.resm_vecl):
            ckpt_plt = torch.load(args.resm_plt)
            pre_dic_plt = ckpt_plt['state_dict']

            model_dic_plt = model_plt.state_dict()
            pre_dic_plt = {k: v for k, v in pre_dic_plt.items() if
                           k in model_dic_plt and model_dic_plt[k].size() == v.size()}
            model_dic_plt.update(pre_dic_plt)
            model_plt.load_state_dict(model_dic_plt)
            args.start_epoch_plt = ckpt_plt['epoch']
            rank1_plt = ckpt_plt['rank1']

            ckpt_vecl = torch.load(args.resm_vecl)
            pre_dic_vecl = ckpt_vecl['state_dict']

            model_dic_vecl = model_vecl.state_dict()
            pre_dic_vecl = {k: v for k, v in pre_dic_vecl.items() if
                           k in model_dic_vecl and model_dic_vecl[k].size() == v.size()}
            model_dic_vecl.update(pre_dic_vecl)
            model_vecl.load_state_dict(model_dic_vecl)
            args.start_epoch_vecl = ckpt_vecl['epoch']
            rank1_vecl = ckpt_vecl['rank1']

            print("\nLoaded checkpoint from '{}' \nand '{}".format(args.resm_plt, args.resm_vecl))
            print("Plate model: start_epoch: {}, rank1: {}".format(args.start_epoch_plt, rank1_plt))
            print("Vehicle model: start_epoch: {}, rank1: {}".format(args.start_epoch_vecl, rank1_vecl))

    if use_gpu:
        model_plt = nn.DataParallel(model_plt).cuda()
        model_vecl = nn.DataParallel(model_vecl).cuda()

    if args.evaluate:
        print("\nEvaluate only")
        test(model_plt, model_vecl, queryloader_plt, queryloader_vecl,
             galleryloader_plt, galleryloader_vecl, use_gpu)
        return


def test(model_plt, model_vecl, queryloader_plt, queryloader_vecl, galleryloader_plt,
         galleryloader_vecl, use_gpu, ranks=[1, 5, 10, 20],
         return_distmat=False):

    if use_plt:
        model_plt.eval()
        with torch.no_grad():
            qf_plt, q_pids_plt, q_camids_plt, q_names_plt = [], [], [], []
            for batch_idx, (imgs, pids, camids, names) in enumerate(queryloader_plt[0]):
                if use_gpu: imgs = imgs.cuda()

                features = model_plt(imgs)
                features = features.data.cpu()

                qf_plt.append(features)
                q_pids_plt.extend(pids)
                q_camids_plt.extend(camids)
                q_names_plt.extend(names)

            qf_plt = torch.cat(qf_plt, 0)
            # qf_flip_plt = torch.cat(qf_flip_plt, 0)
            # qf_plt = (qf_plt + qf_flip_plt) / 2.
            q_pids_plt = np.asarray(q_pids_plt)
            q_camids_plt = np.asarray(q_camids_plt)
            q_names_plt = np.asarray(q_names_plt)
            print("\nPlate model: extracted features for query set, obtained {}-by-{} matrix".\
                  format(qf_plt.size(0), qf_plt.size(1)))

            gf_plt, g_pids_plt, g_camids_plt, g_names_plt = [], [], [], []
            for batch_idx, (imgs, pids, camids, names) in enumerate(galleryloader_plt[0]):
                if use_gpu: imgs = imgs.cuda()

                features = model_plt(imgs)
                features = features.data.cpu()

                gf_plt.append(features)
                g_pids_plt.extend(pids)
                g_camids_plt.extend(camids)
                g_names_plt.extend(names)

            gf_plt = torch.cat(gf_plt, 0)
            # gf_flip_plt = torch.cat(gf_flip_plt, 0)
            # gf_plt = (gf_plt + gf_flip_plt) / 2.
            g_pids_plt = np.asarray(g_pids_plt)
            g_camids_plt = np.asarray(g_camids_plt)
            g_names_plt = np.asarray(g_names_plt)
            print("Plate model: extracted features for gallery set, obtained {}-by-{} matrix"
                  .format(gf_plt.size(0), gf_plt.size(1)))

    if not only_use_plt:
        model_vecl.eval()
        with torch.no_grad():
            qf_vecl, q_pids_vecl, q_camids_vecl, q_groups_vecl, q_names_vecl = [], [], [], [], []
            for batch_idx, (imgs, pids, goplbs, _, camids, _, names) in enumerate(queryloader_vecl[0]):
                if use_gpu: imgs = imgs.cuda()

                features = model_vecl(imgs)
                features = features.data.cpu()

                qf_vecl.append(features)
                q_pids_vecl.extend(pids)
                q_camids_vecl.extend(camids)
                q_groups_vecl.extend(goplbs)
                q_names_vecl.extend(names)

            # qf_flip_vecl = []
            # for imgs, _, _, _ in queryloader_vecl[1]:
            #     features = model_vecl(imgs).data.cpu()
            #     features = torch.div(features, features.norm(dim=1, keepdim=True))
            #     qf_flip_vecl.append(features)

            qf_vecl = torch.cat(qf_vecl, 0)
            # qf_flip_vecl = torch.cat(qf_flip_vecl, 0)
            # qf_vecl = (qf_vecl + qf_flip_vecl) / 2.
            q_pids_vecl = np.asarray(q_pids_vecl)
            q_camids_vecl = np.asarray(q_camids_vecl)
            q_groups_vecl = np.asarray(q_groups_vecl)
            q_names_vecl = np.asarray(q_names_vecl)
            print("Vehicle model: extracted features for query set, obtained {}-by-{} matrix".\
                  format(qf_vecl.size(0), qf_vecl.size(1)))

            gf_vecl, g_pids_vecl, g_camids_vecl, g_groups_vecl, g_names_vecl = [], [], [], [], []
            for batch_idx, (imgs, pids, goplbs, _, camids, _, names) in enumerate(galleryloader_vecl[0]):
                if use_gpu: imgs = imgs.cuda()

                features = model_vecl(imgs)
                features = features.data.cpu()

                gf_vecl.append(features)
                g_pids_vecl.extend(pids)
                g_camids_vecl.extend(camids)
                g_groups_vecl.extend(goplbs)
                g_names_vecl.extend(names)

            # gf_flip_vecl = []
            # for imgs, _, _, _ in galleryloader_vecl[1]:
            #     features = model_vecl(imgs).data.cpu()
            #     features = torch.div(features, features.norm(dim=1, keepdim=True))
            #     gf_flip_vecl.append(features)

            gf_vecl = torch.cat(gf_vecl, 0)
            # gf_flip_vecl = torch.cat(gf_flip_vecl, 0)
            # gf_vecl = (gf_vecl + gf_flip_vecl) / 2.
            g_pids_vecl = np.asarray(g_pids_vecl)
            g_camids_vecl = np.asarray(g_camids_vecl)
            g_groups_vecl = np.asarray(g_groups_vecl)
            g_names_vecl = np.asarray(g_names_vecl)
            print("Vehicle model: extracted features for gallery set, obtained {}-by-{} matrix".\
                  format(gf_vecl.size(0), gf_vecl.size(1)))


    def _distmat_wplt_calc(qf_vecl, gf_vecl, qf_plt, gf_plt):
        qf_vecl, gf_vecl = np.array(qf_vecl), np.array(gf_vecl)
        qf_plt, gf_plt = np.array(qf_plt), np.array(gf_plt)

        # q_incl_pos, q_excl_pos, g_incl_pos, g_excl_pos = [], [], [], []
        q_plt_flg, g_plt_flg = [], []
        for i, name in enumerate(q_names_vecl):
            if name in q_names_plt:
                # q_incl_pos.append(i)
                q_plt_flg.append(1)
            else:
                # q_excl_pos.append(i)
                q_plt_flg.append(0)

        for i, name in enumerate(g_names_vecl):
            if name in g_names_plt:
                # g_incl_pos.append(i)
                g_plt_flg.append(1)
            else:
                # g_excl_pos.append(i)
                g_plt_flg.append(0)

        q_plt_flg = np.array(q_plt_flg)[:, np.newaxis]
        g_plt_flg = np.array(g_plt_flg)[np.newaxis, :]

        all_plt_flg = q_plt_flg * g_plt_flg

        qf_new = np.zeros((len(qf_vecl), 2048))
        gf_new = np.zeros((len(gf_vecl), 2048))
        for i, ft in enumerate(qf_vecl):
            if q_names_vecl[i] in q_names_plt:
                qf_plt_add = k_plt * np.squeeze(qf_plt[q_names_plt == q_names_vecl[i]], 0)
                qf_new[i] = np.concatenate((qf_vecl[i], qf_plt_add))
                # qf_new[i] = np.concatenate((np.zeros(1024), qf_plt_add))
                qf_new[i] = qf_new[i] / np.linalg.norm(qf_new[i])
            else:
                qf_new[i] = np.concatenate((qf_vecl[i], np.zeros(1024)))
        for i, ft in enumerate(gf_vecl):
            if g_names_vecl[i] in g_names_plt:
                gf_plt_add = k_plt * np.squeeze(gf_plt[g_names_plt == g_names_vecl[i]], 0)
                gf_new[i] = np.concatenate((gf_vecl[i], gf_plt_add))
                # gf_new[i] = np.concatenate((np.zeros(1024), gf_plt_add))
                gf_new[i] = gf_new[i] / np.linalg.norm(gf_new[i])
            else:
                gf_new[i] = np.concatenate((gf_vecl[i], np.zeros(1024)))

        distmat = np.zeros((len(qf_new), len(gf_new)))
        for i in range(len(qf_new)):
            if (i + 1) % 400 == 0:
                print("Processed {:.2f}%...".format(i / len(qf_new) * 100))
            for j in range(len(gf_new)):
                cur_qf = qf_new[i].copy()
                cur_gf = gf_new[j].copy()
                if all_plt_flg[i, j] == 0:
                    cur_qf[1024:] = 0
                    cur_gf[1024:] = 0
                distmat[i, j] = np.linalg.norm(cur_qf - cur_gf)

        return distmat

    def _distmat_noplt_calc():
        m, n = qf_vecl.size(0), gf_vecl.size(0)

        distmat = torch.pow(qf_vecl, 2).sum(dim=1, keepdim=True).expand(m, n) + \
                  torch.pow(gf_vecl, 2).sum(dim=1, keepdim=True).expand(n, m).t()
        distmat.addmm_(1, -2, qf_vecl, gf_vecl.t())
        distmat = distmat.numpy()
        return distmat

    def _distmat_only_plt():
        m, n = qf_plt.size(0), gf_plt.size(0)

        distmat = torch.pow(qf_plt, 2).sum(dim=1, keepdim=True).expand(m, n) + \
                  torch.pow(gf_plt, 2).sum(dim=1, keepdim=True).expand(n, m).t()
        distmat.addmm_(1, -2, qf_plt, gf_plt.t())
        distmat = distmat.numpy()
        return distmat

    start_time = time.time()
    print("\nStart calculating distmat.")
    if only_use_plt:
        distmat = _distmat_only_plt()
        q_pids = q_pids_plt
        g_pids = g_pids_plt
        q_camids = q_camids_plt
        g_camids = g_camids_plt
    else:
        q_pids = q_pids_vecl
        g_pids = g_pids_vecl
        q_camids = q_camids_vecl
        g_camids = g_camids_vecl
        if use_plt:
            print("Calculating distmat with plates.")
            distmat = _distmat_wplt_calc(qf_vecl, gf_vecl, qf_plt, gf_plt)
        else:
            print("Calculating distmat without plates.")
            distmat = _distmat_noplt_calc()
        print("Distmat calculation done.\n")

    def _kmeans_rerank():
        all_rk_idxes_ori = np.argsort(distmat, axis=1)
        all_rk_idxes = all_rk_idxes_ori.copy()
        num_q, num_g = distmat.shape
        matches = []
        for q_idx in range(num_q):
            order = all_rk_idxes[q_idx]
            group_label = g_groups_vecl.copy()
            group_label = group_label[order]
            q_pid = q_pids_vecl[q_idx]
            q_camid = q_camids_vecl[q_idx]
            remove = (g_pids_vecl[order] == q_pid) & (g_camids_vecl[order] == q_camid)
            keep = np.invert(remove)

            dstmt = distmat[q_idx].copy()
            rk_idxes = all_rk_idxes[q_idx].copy()
            match = (g_pids_vecl[rk_idxes] == q_pid).astype(np.int32)
            rk_idxes = rk_idxes[keep]
            group_label = group_label[keep]
            match = match[keep]

            top_rk_match = match
            # top_rk_match = match[:top_rerk_num]
            top_rk_idxes = rk_idxes
            # top_rk_idxes = rk_idxes[:top_rerk_num]
            group_label = group_label
            # group_label = group_label[:top_rerk_num]
            top_rk_dstmt = dstmt
            # top_rk_dstmt = dstmt[top_rk_idxes]
            top_rk_gf = gf_vecl[top_rk_idxes]

            # all_clsts, _ = KMeans(top_rk_gf, num_clusters)
            all_clsts = []
            for i in range(1, 5):
                all_clsts.append(np.where(group_label == i)[0])
            lens = [len(clsts) for clsts in all_clsts]
            max_len = max(lens)

            # all_grp_rk_idxes = np.zeros((num_clusters, max_len))
            # all_grp_rk_dstmt = np.zeros((num_clusters, max_len))
            all_grp_rk_idxes = np.zeros((num_clusters, max_len))
            all_grp_rk_dstmt = np.zeros((num_clusters, max_len))
            all_grp_match = np.zeros((num_clusters, max_len))

            rerk_idxes = []
            rerk_match = []
            for i, clsts in enumerate(all_clsts):
                grp_rk_idxes = top_rk_idxes[clsts]
                grp_rk_dstmt = top_rk_dstmt[clsts]
                grp_match = top_rk_match[clsts]
                rlt_rk_idxes = np.argsort(grp_rk_dstmt)
                abs_rk_idxes = grp_rk_idxes[rlt_rk_idxes]
                grp_rk_dstmt = grp_rk_dstmt[rlt_rk_idxes]
                grp_match = grp_match[rlt_rk_idxes]

                all_grp_rk_idxes[i, :len(grp_rk_idxes)] = abs_rk_idxes
                all_grp_rk_idxes[i, len(grp_rk_idxes):] = np.inf
                all_grp_rk_dstmt[i, :len(grp_rk_dstmt)] = grp_rk_dstmt
                all_grp_rk_dstmt[i, len(grp_rk_dstmt):] = np.inf
                all_grp_match[i, :len(grp_match)] = grp_match
                all_grp_match[i, len(grp_match):] = np.inf

            for i in range(max_len):
                smln_idxes = all_grp_rk_idxes[:, i]
                smln_dstmt = all_grp_rk_dstmt[:, i]
                smln_match = all_grp_match[:, i]
                rlt_rk_idxes = np.argsort(smln_dstmt)
                abs_rk_idxes = smln_idxes[rlt_rk_idxes]
                abs_rk_match = smln_match[rlt_rk_idxes]
                for idx in abs_rk_idxes:
                    if idx != np.inf:
                        rerk_idxes.append(idx)
                for mth in abs_rk_match:
                    if mth != np.inf:
                        rerk_match.append(mth)

            # all_rk_idxes[q_idx][:top_rerk_num] = np.array(rerk_idxes).astype(np.int32)
            # rk_idxes[:top_rerk_num] = np.array(rerk_idxes).astype(np.int32)
            # rk_idxes = np.array(rerk_idxes).astype(np.int32)
            # match[:top_rerk_num] = np.array(rerk_match).astype(np.int32)
            match = np.array(rerk_match).astype(np.int32)
            matches.append(match)

        return matches


    if vecl_plt_kmeans:
        print("Start reranking using kmeans.")
        matches = _kmeans_rerank()
        print("Rerank done.\n")
    else:
        matches = None

    print("Start computing CMC and mAP")
    cmc, mAP = evaluate(distmat, q_pids, g_pids, q_camids, g_camids,
                        use_metric_cuhk03=args.use_metric_cuhk03, use_cython=False, matches=matches)
    elapsed = round(time.time() - start_time)
    elapsed = str(datetime.timedelta(seconds=elapsed))
    print("Evaluate test data time (h:m:s): {}.".format(elapsed))
    print("Test data results ----------")
    print("temAP: {:.2%}".format(mAP))
    print("CMC curve")
    for r in ranks:
        print("teRank-{:<3}: {:.2%}".format(r, cmc[r - 1]))
    print("------------------")

    if return_distmat:
        return distmat
    return cmc[0]


if __name__ == '__main__':
    main()


