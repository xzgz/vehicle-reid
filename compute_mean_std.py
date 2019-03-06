import os
import scipy.io
import numpy as np
import os.path as osp

import torch
from torch.utils.data import DataLoader

import data_manager
import transforms as T
from data_manager.dataset_loader import ImageDatasetWGL, ImageDatasetWCL


# dataset = 'vehicleid'
# dataset = 'veri776wgl'
dataset = 'veri776wcl'
data_root = '/home/gysj/pytorch-workspace/pytorch-study/data'
batch_size = 1

print("Initializing dataset {}".format(dataset))
dataset = data_manager.init_imgreid_dataset(name=dataset, root=data_root)

# mean_std = scipy.io.loadmat(osp.join(data_root, 'data_mean_std/veri776_train_image_mean_std.mat'))
mean_std = scipy.io.loadmat(osp.join(data_root, 'data_mean_std/veri776wcl_train_image_mean_std.mat'))
# data_mean = mean_std['mean'][0].astype(np.float64)
# data_std = mean_std['std'][0].astype(np.float64)
data_mean = mean_std['mean'][0].astype(np.float32)
data_std = mean_std['std'][0].astype(np.float32)

transform_test = T.Compose([
    # T.Resize((100, 100)),
    # T.ToTensor(),
    T.Normalize(mean=data_mean, std=data_std),
])

# trainloader = DataLoader(ImageDatasetWGL(dataset, data_type='train', transform=transform_test),
#                          batch_size=batch_size)
# queryloader = DataLoader(ImageDatasetWGL(dataset.query, data_type='query', transform=transform_test),
#                          batch_size=batch_size)
# galleryloader = DataLoader(ImageDatasetWGL(dataset.gallery, data_type='gallery', transform=transform_test),
#                            batch_size=batch_size)
# probeloader = DataLoader(ImageDatasetWGL(dataset.probe, data_type='probe', transform=transform_test),
#                          batch_size=batch_size)
trainloader = DataLoader(ImageDatasetWCL(dataset, data_type='train', merge_h=256, merge_w=256,
                                         mean_std=[data_mean, data_std]), batch_size=batch_size)
queryloader = DataLoader(ImageDatasetWCL(dataset.query, data_type='query', merge_h=256, merge_w=256,
                                         mean_std=[data_mean, data_std]), batch_size=batch_size)
galleryloader = DataLoader(ImageDatasetWCL(dataset.gallery, data_type='gallery', merge_h=256, merge_w=256,
                                           mean_std=[data_mean, data_std]), batch_size=batch_size)

# dataloader = queryloader
# dataloader = galleryloader
# dataloader = probeloader
dataloader = trainloader
# save_path = osp.join(data_root, 'data_mean_std/veri776_query_image_mean_std_norm_100.mat')
# save_path = osp.join(data_root, 'data_mean_std/veri776_gallery_image_mean_std.mat')
# save_path = osp.join(data_root, 'data_mean_std/veri776_gallery_image_mean_std_norm_100.mat')
# save_path = osp.join(data_root, 'data_mean_std/veri776_train_image_mean_std_norm_100.mat')
# save_path = osp.join(data_root, 'data_mean_std/vehicleid_query_image_mean_std.mat')
# save_path = osp.join(data_root, 'data_mean_std/vehicleid_probe_image_mean_std.mat')
# save_path = osp.join(data_root, 'data_mean_std/vehicleid_train_image_mean_std.mat')
# save_path = osp.join(data_root, 'data_mean_std/veri776wcl_query_image_mean_std.mat')
# save_path = osp.join(data_root, 'data_mean_std/veri776wcl_gallery_image_mean_std.mat')
# save_path = osp.join(data_root, 'data_mean_std/veri776wcl_train_image_mean_std.mat')
# save_path = osp.join(data_root, 'data_mean_std/veri776wcl_query_image_mean_std_norm_256.mat')
# save_path = osp.join(data_root, 'data_mean_std/veri776wcl_gallery_image_mean_std_norm_256.mat')
save_path = osp.join(data_root, 'data_mean_std/veri776wcl_train_image_mean_std_norm_256.mat')


def compute_mean_std():
    with torch.no_grad():
        std, mean = [], []
        for batch_idx, (imgs, _, _, _, _, _) in enumerate(dataloader):
            imgs = imgs.view(3, -1)
            mean_val = torch.mean(imgs, dim=1).unsqueeze(dim=0)
            std_val = torch.std(imgs, dim=1).unsqueeze(dim=0)
            mean.append(mean_val)
            std.append(std_val)

            if (batch_idx + 1) % 400 == 0:
                print('Read {}/{} batch images.'.format(batch_idx + 1, len(dataloader)))
        mean = torch.cat(mean, 0)
        std = torch.cat(std, 0)
    mean_val = torch.mean(mean, dim=0)
    std_val = torch.mean(std, dim=0)
    result = {'mean': mean_val.numpy(), 'std': std_val.numpy()}
    scipy.io.savemat(save_path, result)
    print(mean_val, std_val)

    # result = scipy.io.loadmat(save_path)
    # print(result['std'], result['mean'])
# compute_mean_std()


def compute_mean_std_wcl():
    with torch.no_grad():
        std, mean = [], []
        for batch_idx, (imgs, _, _, _, _, _) in enumerate(dataloader):
            # rgb = imgs[0].view(-1, 3)
            # color = imgs[1].view(-1, 1)
            # lbp = imgs[2].view(-1, 1)
            # mean_rgb = torch.mean(rgb, dim=0)
            # mean_color = torch.mean(color, dim=0)
            # mean_lbp = torch.mean(lbp, dim=0)
            # std_rgb = torch.std(rgb, dim=0)
            # std_color = torch.std(color, dim=0)
            # std_lbp = torch.std(lbp, dim=0)
            #
            # mean.append(torch.cat([mean_rgb, mean_color, mean_lbp], dim=0).unsqueeze(dim=0))
            # std.append(torch.cat([std_rgb, std_color, std_lbp], dim=0).unsqueeze(dim=0))

            imgs = imgs.view(5, -1)
            mean_val = torch.mean(imgs, dim=1)
            std_val = torch.std(imgs, dim=1)
            mean.append(mean_val.unsqueeze(dim=0))
            std.append(std_val.unsqueeze(dim=0))

            if (batch_idx + 1) % 400 == 0:
                print('Read {}/{} batch images.'.format(batch_idx + 1, len(dataloader)))
        mean = torch.cat(mean, 0)
        std = torch.cat(std, 0)
    mean_val = torch.mean(mean, dim=0)
    std_val = torch.mean(std, dim=0)
    result = {'mean': mean_val.numpy(), 'std': std_val.numpy()}
    scipy.io.savemat(save_path, result)
    print(mean_val, std_val)
compute_mean_std_wcl()


def show_mean_std():
    mean_std_mat_dir = osp.join(data_root, 'data_mean_std')
    for root, _, files in os.walk(mean_std_mat_dir):
        for file in files:
            mean_std_mat_file = osp.join(root, file)
            result = scipy.io.loadmat(mean_std_mat_file)
            print(file, 'mean:', result['mean'], 'std:', result['std'])
# show_mean_std()


# def compute_mean_std():
#     # dataloader = queryloader
#     dataloader = galleryloader
#     with torch.no_grad():
#         qf = []
#         for batch_idx, (imgs, _, _, _, _, _) in enumerate(dataloader):
#             qf.append(imgs)
#             if (batch_idx + 1) % 10 == 0:
#                 print('Read {}/{} batch images.'.format(batch_idx + 1, len(dataloader)))
#         qf = torch.cat(qf, 0)
#     print(qf.size())
#     size_origin = qf.size()
#     qf = torch.transpose(qf, dim0=3, dim1=1)
#     qf = torch.transpose(qf, dim0=1, dim1=2).contiguous()
#     qf = qf.view(-1, size_origin[1])
#     std_val = torch.std(qf, dim=0)
#     mean_val = torch.mean(qf, dim=0)
#     result = {'mean': mean_val.numpy(), 'std': std_val.numpy()}
#     scipy.io.savemat(save_path, result)
#     print(std_val, mean_val)
#
#     # result = scipy.io.loadmat(save_path)
#     # print(result['std'], result['mean'])



