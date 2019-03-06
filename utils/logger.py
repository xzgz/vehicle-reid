from __future__ import absolute_import

import sys
import os
import re
import os.path as osp

from .iotools import mkdir_if_missing


class Logger(object):
    """
    Write console output to external text file.
    Code imported from https://github.com/Cysu/open-reid/blob/master/reid/utils/logging.py.
    """
    def __init__(self, save_dir, checkpoint_suffix=None, evaluate=False):
        self.console = sys.stdout
        self.file = None
        self.checkpoint_suffix = checkpoint_suffix
        self.start_epoch, self.train_step = self.get_start_epoch()

        current_log_file = osp.join(save_dir, 'log_train' + self.checkpoint_suffix + '.txt')
        previous_log_file = 'log_train'
        if self.train_step >= 1 and not evaluate:
            if self.train_step == 1:
                previous_log_file = osp.join(save_dir, 'log_train.txt')
            else:
                suffix_list = self.checkpoint_suffix.split('_')
                for i in range(1, len(suffix_list) - 1):
                    previous_log_file += '_'
                    previous_log_file += suffix_list[i]
                previous_log_file += '.txt'
            previous_log_file = osp.join(save_dir, previous_log_file)
            previous_log = self.get_previous_log(previous_log_file)
            self.init_current_log_file(current_log_file, previous_log)

        if evaluate:
            fpath = osp.join(save_dir, 'log_test.txt')
        else:
            fpath = current_log_file

        mkdir_if_missing(osp.dirname(fpath))
        if evaluate or self.train_step < 1:
            self.file = open(fpath, 'w')
        else:
            self.file = open(fpath, 'a')

    def __del__(self):
        self.close()

    def __enter__(self):
        pass

    def __exit__(self, *args):
        self.close()

    def write(self, msg):
        self.console.write(msg)
        if self.file is not None:
            self.file.write(msg)

    def flush(self):
        self.console.flush()
        if self.file is not None:
            self.file.flush()
            os.fsync(self.file.fileno())

    def close(self):
        self.console.close()
        if self.file is not None:
            self.file.close()

    def get_start_epoch(self):
        pattern_start_epoch = re.compile('_sf(\d+)')
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




