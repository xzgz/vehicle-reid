from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path as osp
from torchvision.datasets import ImageFolder


class VehicleID(object):

    dataset_dir = 'VehicleID_V1.0'

    def __init__(self, root='data', test_size='large', verbose=True, **kwargs):
        super(VehicleID, self).__init__()
        self.dataset_dir = osp.join(root, self.dataset_dir)
        if test_size not in ['small', 'medium', 'large']:
            raise KeyError('Unsupported test size: {}'.format(test_size))

        test_size_path_dict = {'small': 'vehicleid-small', 'medium': 'vehicleid-medium', 'large': 'vehicleid-large'}
        self.train_dir = osp.join(self.dataset_dir, 'train_all')
        self.query_dir = osp.join(self.dataset_dir, test_size_path_dict[test_size], 'query')
        self.gallery_dir = osp.join(self.dataset_dir, test_size_path_dict[test_size], 'gallery')
        self.probe_dir = osp.join(self.dataset_dir, test_size_path_dict[test_size], 'probe')

        self._check_before_run()

        train, num_train_pids, num_train_imgs = self._process_dir(self.train_dir)
        query, num_query_pids, num_query_imgs = self._process_dir(self.query_dir)
        gallery, num_gallery_pids, num_gallery_imgs = self._process_dir(self.gallery_dir)
        probe, num_probe_pids, num_probe_imgs = self._process_dir(self.probe_dir)

        num_total_pids = num_train_pids + num_query_pids
        num_total_imgs = num_train_imgs + num_query_imgs + num_gallery_imgs

        if verbose:
            print("=> VehicleID loaded")
            print("Dataset statistics:")
            print("  ------------------------------")
            print("  subset        | # ids | # images")
            print("  ------------------------------")
            print("  train         | {:5d} | {:8d}".format(num_train_pids, num_train_imgs))
            print("  query         | {:5d} | {:8d}".format(num_query_pids, num_query_imgs))
            print("  gallery       | {:5d} | {:8d}".format(num_gallery_pids, num_gallery_imgs))
            print("  probe         | {:5d} | {:8d}".format(num_probe_pids, num_probe_imgs))
            print("  ------------------------------")
            print("  total         | {:5d} | {:8d}".format(num_total_pids, num_total_imgs))
            print("  ------------------------------")

        self.train = train
        self.query = query
        self.gallery = gallery
        self.probe = probe

        self.num_train_pids = num_train_pids
        self.num_query_pids = num_query_pids
        self.num_gallery_pids = num_gallery_pids

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))
        if not osp.exists(self.query_dir):
            raise RuntimeError("'{}' is not available".format(self.query_dir))
        if not osp.exists(self.gallery_dir):
            raise RuntimeError("'{}' is not available".format(self.gallery_dir))

    def _process_dir(self, dir_path):
        img_path_vid = ImageFolder(dir_path).imgs
        pid_container = set()
        dataset = []
        camid = 0
        gopid = 0

        for i, pth_vid in enumerate(img_path_vid):
            pid_container.add(pth_vid[1])
            dataset.append([i, pth_vid[1], camid, gopid, pth_vid[0]])

        num_pids = len(pid_container)
        num_imgs = len(dataset)
        return dataset, num_pids, num_imgs



