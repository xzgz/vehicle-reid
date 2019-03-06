from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import glob
import re
import os.path as osp


def list_pictures(directory, ext='jpg|jpeg|bmp|png|ppm'):
    return sorted([os.path.join(root, f)
                   for root, _, files in os.walk(directory) for f in files
                   if re.match(r'([\w]+\.(?:' + ext + '))', f)])


def list_picture_name(directory, ext='jpg|jpeg|bmp|png|ppm'):
    return sorted([f for root, _, files in os.walk(directory) for f in files
                   if re.match(r'([\w]+\.(?:' + ext + '))', f)])


class VeRi776(object):

    dataset_dir = 'VeRi'

    def __init__(self, root='data', verbose=True, **kwargs):
        super(VeRi776, self).__init__()
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.train_dir = osp.join(self.dataset_dir, 'image_train')
        self.query_dir = osp.join(self.dataset_dir, 'image_query')
        self.gallery_dir = osp.join(self.dataset_dir, 'image_test')
        self.train_gallery_dir = osp.join(self.dataset_dir, 'gallery_from_train_eq3')
        self.train_query_dir = osp.join(self.dataset_dir, 'query_from_train_v2')

        self._check_before_run()

        train, num_train_pids, num_train_imgs = self._process_dir(self.train_dir, relabel=True)
        query, num_query_pids, num_query_imgs = self._process_dir(self.query_dir, relabel=False)
        gallery, num_gallery_pids, num_gallery_imgs = self._process_dir(self.gallery_dir, relabel=False)
        train_query, num_train_query_pids, num_train_query_imgs = self._process_dir(self.train_query_dir,
                                                                                    relabel=False)
        train_gallery, num_train_gallery_pids, num_train_gallery_imgs = self._process_dir(self.train_gallery_dir,
                                                                                          relabel=False)
        num_total_pids = num_train_pids + num_query_pids
        num_total_imgs = num_train_imgs + num_query_imgs + num_gallery_imgs

        if verbose:
            print("=> VeRi776 loaded")
            print("Dataset statistics:")
            print("  ------------------------------")
            print("  subset        | # ids | # images")
            print("  ------------------------------")
            print("  train         | {:5d} | {:8d}".format(num_train_pids, num_train_imgs))
            print("  query         | {:5d} | {:8d}".format(num_query_pids, num_query_imgs))
            print("  gallery       | {:5d} | {:8d}".format(num_gallery_pids, num_gallery_imgs))
            print("  train_query   | {:5d} | {:8d}".format(num_train_query_pids, num_train_query_imgs))
            print("  train_gallery | {:5d} | {:8d}".format(num_train_gallery_pids, num_train_gallery_imgs))
            print("  ------------------------------")
            print("  total    | {:5d} | {:8d}".format(num_total_pids, num_total_imgs))
            print("  ------------------------------")

        self.train = train
        self.query = query
        self.gallery = gallery
        self.train_query = train_query
        self.train_gallery = train_gallery

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

    def _process_dir(self, dir_path, relabel=False):
        # img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
        img_paths = list_pictures(dir_path)
        pattern = re.compile(r'([-\d]+)_c([\d]+)')

        pid_container = set()
        for img_path in img_paths:
            pid, _ = map(int, pattern.search(img_path).groups())
            if pid == -1: continue  # junk images are just ignored
            pid_container.add(pid)
        pid2label = {pid:label for label, pid in enumerate(pid_container)}

        dataset = []
        for img_path in img_paths:
            pid, camid = map(int, pattern.search(img_path).groups())
            if pid == -1: continue  # junk images are just ignored
            assert 1 <= pid <= 776  # pid == 0 means background
            assert 1 <= camid <= 20
            # camid -= 1 # index starts from 0
            if relabel: pid = pid2label[pid]
            dataset.append((img_path, pid, camid))

        num_pids = len(pid_container)
        num_imgs = len(dataset)
        return dataset, num_pids, num_imgs


class VeRi776Plt(object):

    dataset_dir = 'VeRi-plate'

    def __init__(self, root='data', verbose=True, **kwargs):
        super(VeRi776Plt, self).__init__()
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.train_dir = osp.join(self.dataset_dir, 'plate_train')
        self.query_dir = osp.join(self.dataset_dir, 'plate_query')
        self.gallery_dir = osp.join(self.dataset_dir, 'plate_test')

        self._check_before_run()

        train, num_train_pids, num_train_imgs = self._process_dir(self.train_dir, relabel=True)
        query, num_query_pids, num_query_imgs = self._process_dir(self.query_dir, relabel=False)
        gallery, num_gallery_pids, num_gallery_imgs = self._process_dir(self.gallery_dir, relabel=False)

        num_total_pids = num_train_pids + num_query_pids
        num_total_imgs = num_train_imgs + num_query_imgs + num_gallery_imgs

        if verbose:
            print("=> VeRi776 loaded")
            print("Dataset statistics:")
            print("  ------------------------------")
            print("  subset        | # ids | # images")
            print("  ------------------------------")
            print("  train         | {:5d} | {:8d}".format(num_train_pids, num_train_imgs))
            print("  query         | {:5d} | {:8d}".format(num_query_pids, num_query_imgs))
            print("  gallery       | {:5d} | {:8d}".format(num_gallery_pids, num_gallery_imgs))
            print("  ------------------------------")
            print("  total    | {:5d} | {:8d}".format(num_total_pids, num_total_imgs))
            print("  ------------------------------")

        self.train = train
        self.query = query
        self.gallery = gallery

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

    def _process_dir(self, dir_path, relabel=False):
        # img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
        img_paths = list_pictures(dir_path)
        pattern = re.compile(r'([-\d]+)_c([\d]+)')

        pid_container = set()
        for img_path in img_paths:
            pid, _ = map(int, pattern.search(img_path).groups())
            if pid == -1: continue  # junk images are just ignored
            pid_container.add(pid)
        pid2label = {pid:label for label, pid in enumerate(pid_container)}

        dataset = []
        for img_path in img_paths:
            pid, camid = map(int, pattern.search(img_path).groups())
            if pid == -1: continue  # junk images are just ignored
            assert 1 <= pid <= 776  # pid == 0 means background
            assert 1 <= camid <= 20
            # camid -= 1 # index starts from 0
            if relabel: pid = pid2label[pid]
            dataset.append((img_path, pid, camid))

        num_pids = len(pid_container)
        num_imgs = len(dataset)
        return dataset, num_pids, num_imgs


class VeRi776WithGroupLabel(object):

    dataset_dir = 'VeRi'

    def __init__(self, root='data', verbose=True, **kwargs):
        super(VeRi776WithGroupLabel, self).__init__()
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.train_dir = osp.join(self.dataset_dir, 'image_train')
        self.query_dir = osp.join(self.dataset_dir, 'image_query')
        self.gallery_dir = osp.join(self.dataset_dir, 'image_test')
        self.train_gallery_dir = osp.join(self.dataset_dir, 'gallery_from_train_eq3')
        self.train_query_dir = osp.join(self.dataset_dir, 'query_from_train_v2')
        self.name_query_with_view = osp.join(self.dataset_dir, 'name_query_with_view.txt')
        self.name_test_with_view = osp.join(self.dataset_dir, 'name_test_with_view.txt')
        self.name_train_with_view = osp.join(self.dataset_dir, 'name_train_with_view.txt')

        self._check_before_run()

        # train, num_train_pids, num_train_imgs = \
        #     self._process_dir(self.train_dir, relabel=True, group_label=True, data_type='train')
        # query, num_query_pids, num_query_imgs = \
        #     self._process_dir(self.query_dir, relabel=False, group_label=True, data_type='query')
        # gallery, num_gallery_pids, num_gallery_imgs = \
        #     self._process_dir(self.gallery_dir, relabel=False, group_label=True, data_type='gallery')
        train, num_train_pids, num_train_imgs = \
            self._process_dir(self.train_dir, relabel=True, group_label=False, data_type='train')
        query, num_query_pids, num_query_imgs = \
            self._process_dir(self.query_dir, relabel=False, group_label=False, data_type='query')
        gallery, num_gallery_pids, num_gallery_imgs = \
            self._process_dir(self.gallery_dir, relabel=False, group_label=False, data_type='gallery')

        train_query, num_train_query_pids, num_train_query_imgs = \
            self._process_dir(self.train_query_dir, relabel=False)
        train_gallery, num_train_gallery_pids, num_train_gallery_imgs = \
            self._process_dir(self.train_gallery_dir, relabel=False)
        num_total_pids = num_train_pids + num_query_pids
        num_total_imgs = num_train_imgs + num_query_imgs + num_gallery_imgs

        if verbose:
            print("=> VeRi776 loaded")
            print("Dataset statistics:")
            print("  ------------------------------")
            print("  subset        | # ids | # images")
            print("  ------------------------------")
            print("  train         | {:5d} | {:8d}".format(num_train_pids, num_train_imgs))
            print("  query         | {:5d} | {:8d}".format(num_query_pids, num_query_imgs))
            print("  gallery       | {:5d} | {:8d}".format(num_gallery_pids, num_gallery_imgs))
            print("  train_query   | {:5d} | {:8d}".format(num_train_query_pids, num_train_query_imgs))
            print("  train_gallery | {:5d} | {:8d}".format(num_train_gallery_pids, num_train_gallery_imgs))
            print("  ------------------------------")
            print("  total    | {:5d} | {:8d}".format(num_total_pids, num_total_imgs))
            print("  ------------------------------")

        self.train = train
        self.query = query
        self.gallery = gallery
        self.train_query = train_query
        self.train_gallery = train_gallery

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

    def _process_dir(self, dir_path, relabel=False, group_label=False, data_type=None):
        # img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
        # pattern = re.compile(r'([-\d]+)_c([\d]+)')
        pid_container = set()
        img_id = 0
        gopid = 0
        dataset = {}

        if group_label:
            if data_type == 'train':
                name_txt = self.name_train_with_view
            elif data_type == 'query':
                name_txt = self.name_query_with_view
            else:
                name_txt = self.name_test_with_view
            f_txt = open(name_txt)
            img_paths = f_txt.readlines()
            f_txt.close()
            pattern = re.compile('c([\d]+)_([\d]+)_([\d]+)_([\d]+)_([\d]+)_([\d]+)_([\d]+)')

            for img_path in img_paths:
                img_path = img_path.split()
                img_path = [im_pth.strip() for im_pth in img_path]
                _, _, _, _, pid, _, _ = map(int, pattern.search(img_path[1]).groups())
                if pid == -1: continue  # junk images are just ignored
                pid_container.add(pid)
            pid2label = {pid: label for label, pid in enumerate(pid_container)}

            for img_path in img_paths:
                img_path = img_path.split()
                img_path = [im_pth.strip() for im_pth in img_path]
                camid, gopid, _, _, pid, _, _ = map(int, pattern.search(img_path[1]).groups())
                if pid == -1: continue  # junk images are just ignored
                assert 1 <= pid <= 776  # pid == 0 means background
                assert 1 <= camid <= 20
                if relabel: pid = pid2label[pid]
                img_path = osp.join(dir_path, img_path[0])
                dataset[img_id] = [img_id, pid, camid, gopid, img_path]
                img_id += 1
        else:
            img_paths = list_pictures(dir_path)
            pattern = re.compile('([\d]+)_c([\d]+)')

            for img_path in img_paths:
                pid, _ = map(int, pattern.search(img_path).groups())
                if pid == -1: continue  # junk images are just ignored
                pid_container.add(pid)
            pid2label = {pid: label for label, pid in enumerate(pid_container)}

            for img_path in img_paths:
                pid, camid = map(int, pattern.search(img_path).groups())
                if pid == -1: continue  # junk images are just ignored
                assert 1 <= pid <= 776  # pid == 0 means background
                assert 1 <= camid <= 20
                # camid -= 1  # index starts from 0
                if relabel: pid = pid2label[pid]
                dataset[img_id] = [img_id, pid, camid, gopid, img_path]
                img_id += 1

        num_pids = len(pid_container)
        num_imgs = len(dataset)
        return dataset, num_pids, num_imgs

    def _update_group_label(self, group_label):
        for gl in group_label:
            self.train[gl[0]][3] = gl[1]


class VeRi776WithColorLBP(object):

    dataset_dir = 'VeRi'

    def __init__(self, root='data', verbose=True, **kwargs):
        super(VeRi776WithColorLBP, self).__init__()
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.train_dir = osp.join(self.dataset_dir, 'image_train')
        self.query_dir = osp.join(self.dataset_dir, 'image_query')
        self.gallery_dir = osp.join(self.dataset_dir, 'image_test')
        self.train_gallery_dir = osp.join(self.dataset_dir, 'gallery_from_train_eq3')
        self.train_query_dir = osp.join(self.dataset_dir, 'query_from_train_v2')
        self.train_rgb_color_lbp_dir = osp.join(self.dataset_dir, 'image_train_re_arrange')
        self.test_rgb_color_lbp_dir = osp.join(self.dataset_dir, 'image_test_re_arrange')
        self.name_test_new2origin = osp.join(self.dataset_dir, 'name_test_new2origin.txt')
        self.name_train_new2origin = osp.join(self.dataset_dir, 'name_train_new2origin.txt')

        self._check_before_run()

        train, num_train_pids, num_train_imgs = \
            self._process_dir(self.train_dir, relabel=True, data_source='train')
        query, num_query_pids, num_query_imgs = \
            self._process_dir(self.query_dir, relabel=False, data_source='test')
        gallery, num_gallery_pids, num_gallery_imgs = \
            self._process_dir(self.gallery_dir, relabel=False, data_source='test')

        train_query, num_train_query_pids, num_train_query_imgs = \
            self._process_dir(self.train_query_dir, relabel=False, data_source='train')
        train_gallery, num_train_gallery_pids, num_train_gallery_imgs = \
            self._process_dir(self.train_gallery_dir, relabel=False, data_source='train')
        num_total_pids = num_train_pids + num_query_pids
        num_total_imgs = num_train_imgs + num_query_imgs + num_gallery_imgs

        if verbose:
            print("=> VeRi776 loaded")
            print("Dataset statistics:")
            print("  ------------------------------")
            print("  subset        | # ids | # images")
            print("  ------------------------------")
            print("  train         | {:5d} | {:8d}".format(num_train_pids, num_train_imgs))
            print("  query         | {:5d} | {:8d}".format(num_query_pids, num_query_imgs))
            print("  gallery       | {:5d} | {:8d}".format(num_gallery_pids, num_gallery_imgs))
            print("  train_query   | {:5d} | {:8d}".format(num_train_query_pids, num_train_query_imgs))
            print("  train_gallery | {:5d} | {:8d}".format(num_train_gallery_pids, num_train_gallery_imgs))
            print("  ------------------------------")
            print("  total    | {:5d} | {:8d}".format(num_total_pids, num_total_imgs))
            print("  ------------------------------")

        self.train = train
        self.query = query
        self.gallery = gallery
        self.train_query = train_query
        self.train_gallery = train_gallery

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

    def _process_dir(self, name_dir, relabel=False, group_label=False, data_source='train'):
        pid_container = set()
        img_id = 0
        dataset = {}
        img_name = list_picture_name(name_dir)
        pattern = re.compile('([\d]+)_c([\d]+)')
        name2path = {}

        if data_source == 'train':
            name2path_file = self.name_train_new2origin
            data_dir = self.train_rgb_color_lbp_dir
        else:
            name2path_file = self.name_test_new2origin
            data_dir = self.test_rgb_color_lbp_dir
        name2path_f = open(name2path_file)
        name2path_list = name2path_f.readlines()
        name2path_f.close()
        for n2p in name2path_list:
            n2p = n2p.strip().split()
            name2path[n2p[2]] = n2p[0] + '/' + n2p[1]

        for name in img_name:
            pid, _ = map(int, pattern.search(name).groups())
            if pid == -1: continue  # junk images are just ignored
            pid_container.add(pid)
        if relabel:
            pid2label = {pid: label for label, pid in enumerate(pid_container)}

        for name in img_name:
            pid, camid = map(int, pattern.search(name).groups())
            if pid == -1: continue  # junk images are just ignored
            assert 1 <= pid <= 776  # pid == 0 means background
            assert 1 <= camid <= 20
            # camid -= 1  # index starts from 0
            if relabel: pid = pid2label[pid]
            origin_name = name
            img_rgb_path = osp.join(data_dir, name2path[name] + '.jpg')
            img_color_lbp_path = osp.join(data_dir, name2path[name] + '.mat')
            dataset[img_id] = [img_id, pid, camid, origin_name, img_rgb_path, img_color_lbp_path]
            img_id += 1

        num_pids = len(pid_container)
        num_imgs = len(dataset)
        return dataset, num_pids, num_imgs



