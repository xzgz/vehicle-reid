from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from PIL import Image
import numpy as np
import os.path as osp
import scipy.io
import cv2

import torch
from torch.utils.data import Dataset
from torch.utils.data.sampler import BatchSampler


def read_image(img_path):
    """Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process."""
    got_img = False
    if not osp.exists(img_path):
        raise IOError("{} does not exist".format(img_path))
    while not got_img:
        try:
            img = Image.open(img_path).convert('RGB')
            got_img = True
        except IOError:
            print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
            pass
    return img


class ImageDataset(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_path, pid, camid = self.dataset[index]
        img = read_image(img_path)
        
        if self.transform is not None:
            img = self.transform(img)
        
        return img, pid, camid, img_path


class ImageDatasetV2(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_path, pid, camid = self.dataset[index]
        img = read_image(img_path)

        if self.transform is not None:
            img = self.transform(img)
        img_name = osp.basename(img_path)

        return img, pid, camid, img_name


class ImageDatasetWGL(Dataset):
    def __init__(self, dataset, data_type=None, transform=None, with_image_name=False):
        if data_type == 'train':
            self.dataset = dataset.train
            self.dataset_op = dataset
        else:
            self.dataset = dataset
            self.dataset_op = None
        self.transform = transform
        self.with_image_name = with_image_name

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_id, pid, camid, group_label, img_path = self.dataset[index]
        img = read_image(img_path)

        if self.transform is not None:
            img = self.transform(img)
        if self.with_image_name:
            img_name = osp.basename(img_path)
            return img, pid, group_label, img_id, camid, img_path, img_name
        else:
            return img, pid, group_label, img_id, camid, img_path


class ImageDatasetWCL(Dataset):
    def __init__(self, dataset, data_type=None, merge_h=256, merge_w=256,
                 mean_std=None, with_image_name=False):
        if data_type == 'train':
            self.dataset = dataset.train
            self.dataset_op = dataset
        else:
            self.dataset = dataset
            self.dataset_op = None
        self.merge_h = merge_h
        self.merge_w = merge_w
        self.mean_std = mean_std
        self.with_image_name = with_image_name
        self.group_label = 0

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_id, pid, camid, origin_name, img_rgb_path, img_color_lbp_path = self.dataset[index]
        img_rgb = cv2.imread(img_rgb_path)
        img_rgb = img_rgb.astype(np.float32) / 255.0
        img_color_lbp = scipy.io.loadmat(img_color_lbp_path)
        img_color = img_color_lbp['feature_color'].astype(np.float32) / 588.0
        # img_lbp = img_color_lbp['feature_LBP'].astype(np.float32) / 255.0

        img_rgb = cv2.resize(img_rgb, (self.merge_w, self.merge_h))
        img_color = cv2.resize(img_color, (self.merge_w, self.merge_h))
        # img_lbp = cv2.resize(img_lbp, (self.merge_w, self.merge_h))

        img_color = img_color[:, :, np.newaxis]
        # img_lbp = img_lbp[:, :, np.newaxis]
        # img = np.concatenate([img_rgb, img_color, img_lbp], axis=2).transpose([2, 0, 1])
        img = np.concatenate([img_rgb, img_color], axis=2).transpose([2, 0, 1])
        if self.mean_std is not None:
            img = (img - self.mean_std[0][:, np.newaxis, np.newaxis]) / self.mean_std[1][:, np.newaxis, np.newaxis]
        # img = [img_rgb, img_color, img_lbp]

        if self.with_image_name:
            return img, pid, self.group_label, img_id, camid, img_rgb_path, origin_name
        else:
            return img, pid, self.group_label, img_id, camid, img_rgb_path


class ClassSampler(BatchSampler):
    def __init__(self, dataset, sample_cls_cnt, each_cls_cnt):
        if isinstance(dataset, dict):
            dataset = list(dataset.values())
        self.dataset = dataset
        self.dataset_len = len(dataset)
        self.class_label = np.array(dataset)[:, 1].astype(np.int32)
        self.class_set = list(set(self.class_label))
        np.random.shuffle(self.class_set)
        self.class_count = len(self.class_set)
        self.class_to_indices = {cls: np.where(self.class_label == cls)[0]
                                 for cls in self.class_set}

        self.sample_cls_cnt = sample_cls_cnt
        self.each_cls_cnt = each_cls_cnt
        self.batch_size = self.sample_cls_cnt * self.each_cls_cnt
        # self.batch_number = self.dataset_len // self.batch_size * 6
        self.batch_number = self.dataset_len // self.batch_size
        self.used_each_class_count = {cls: 0 for cls in self.class_set}
        self.used_class_count = 0
        self.batch_count = 0

    def get_less_class_indices(self, class_indices, count):
        chosen_indices = []
        cls_len = len(class_indices)
        cyc_cnt = count // cls_len
        rem_cnt = count - cyc_cnt * cls_len
        chosen_indices.extend(np.tile(class_indices, cyc_cnt))
        chosen_indices.extend(class_indices[:rem_cnt])
        return chosen_indices

    def __iter__(self):
        self.batch_count = 0
        for cls in self.class_set:
            np.random.shuffle(self.class_to_indices[cls])

        while self.batch_count < self.batch_number:
            if self.used_class_count + self.sample_cls_cnt < self.class_count:
                class_chosen = self.class_set[self.used_class_count:self.used_class_count + self.sample_cls_cnt]
            else:
                remained_count = self.used_class_count + self.sample_cls_cnt - self.class_count
                class_chosen = self.class_set[self.used_class_count:]
                class_chosen.extend(self.class_set[:remained_count])
                np.random.shuffle(self.class_set)
                self.used_class_count = 0

            indices = []
            for cls in class_chosen:
                if self.each_cls_cnt > len(self.class_to_indices[cls]):
                    indices.extend(self.get_less_class_indices(self.class_to_indices[cls],
                                                               self.each_cls_cnt))
                else:
                    if self.used_each_class_count[cls] + self.each_cls_cnt < len(self.class_to_indices[cls]):
                        indices.extend(self.class_to_indices[cls][self.used_each_class_count[cls]:
                                       self.used_each_class_count[cls] + self.each_cls_cnt])
                        self.used_each_class_count[cls] += self.each_cls_cnt
                    else:
                        remained_count = self.used_each_class_count[cls] + self.each_cls_cnt \
                                         - len(self.class_to_indices[cls])
                        indices.extend(self.class_to_indices[cls][self.used_each_class_count[cls]:])
                        indices.extend(self.class_to_indices[cls][:remained_count])
                        self.used_each_class_count[cls] = remained_count

            self.used_class_count += self.sample_cls_cnt
            self.batch_count += 1

            yield indices

    def __len__(self):
        return self.batch_number


