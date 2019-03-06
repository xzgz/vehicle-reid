import re
import os
import numpy as np
import matplotlib.pyplot as plt


version_add_evaluate_train_data = True
log_dir = '/media/sda1/sleep-data/gysj/log-reid/train-hyg'
# checkpoint_suffix = '_sf52lr1ef4'

log_param_list = [
                  'vggm1024v3pre-nolrn-gstev2-did512-bh60-lr1ef2',
                  'vggm1024v3pre-xtlv2-bbs-bh60-lr1ef2',
                  'vggm1024v3pre-xtl-bbs-bh60-lr1ef2',
                  'vggm1024v3pre-xent-cclbs-bh60-lr1ef2',
                  # 'vggm1024v2pre-xent-ccl-veri776-bh60-lr1ef2-mxe90',
                  # 'vggm1024v2pre-gstev2-no-intra-bh60-lr1ef2',
                  # 'vggm1024v2pre-gstev2-bh60-lr1ef2',
                  # 'vggm1024v2pre-gste-small-batch-lr1ef2',
                  # 'vggm1024v2pre-gste-veri776-bh60-lr1ef2',
                  # 'vggm1024v2pre-anglem1-veri776-bh60-lr1ef2',
                  # 'vggm1024v2pre-anglem4-veri776-bh60-lr1ef2',
                  # 'vggm1024v2pre-xent-veri776-bh56-lr1ef2',
                  # 'vggm1024v2pre-xent-ccl-veri776-bh60-lr1ef2-maxepoch90',
                  # 'vggm1024v2pre-xent-ccl-veri776-bh60-lr1ef2',
                  # 'vggm1024v2pre-xent-cclv2-veri776-bh60-lr1ef2',
                  # 'vggm1024v2pre-xent-trtrnt-veri776-bh60-lr1ef2',
                 ]
suffix_model_list = [
                     # 'vggm1024-tripletrnt-veri776-bh64-lr1ef4',
                     # 'vggm1024-tripletv2rnt-veri776-bh64-lr1ef3',
                     # 'hacnn-xent-market1501-bh64-lr1ef2',
                    ]

pattern_eval_step = re.compile('eval_step=(\d+)')
pattern_print_freq = re.compile('print_freq=(\d+)')
pattern_batch_count = re.compile('\[\d+\]\[\d+/(\d+)\]')
pattern_best_rank1 = re.compile('Best\sRank-1\s(\d+.\d+)')
pattern_btrk1_epoch = re.compile('achieved\sat\sepoch\s(\d+)')
pattern_loss = re.compile('Loss\s(\d+.\d+)\s.(\d+.\d+).')
pattern_map = re.compile('temAP:\s(\d+.\d+)')
pattern_cmc1 = re.compile('teRank-1\s\s:\s(\d+.\d+)')
pattern_cmc5 = re.compile('teRank-5\s\s:\s(\d+.\d+)')
pattern_cmc10 = re.compile('teRank-10\s:\s(\d+.\d+)')
pattern_cmc20 = re.compile('teRank-20\s:\s(\d+.\d+)')
pattern_tmap = re.compile('trmAP:\s(\d+.\d+)')
pattern_tcmc1 = re.compile('trRank-1\s\s:\s(\d+.\d+)')
pattern_tcmc5 = re.compile('trRank-5\s\s:\s(\d+.\d+)')
pattern_tcmc10 = re.compile('trRank-10\s:\s(\d+.\d+)')
pattern_tcmc20 = re.compile('trRank-20\s:\s(\d+.\d+)')

def get_train_info():
    train_info_dict = {}
    for param in log_param_list:
        if param in suffix_model_list:
            log_file = '/media/sda1/sleep-data/gysj/log-reid/train-hyg/' + param + '/log_train' \
                       + checkpoint_suffix + '.txt'
        else:
            log_file = '/media/sda1/sleep-data/gysj/log-reid/train-hyg/' + param + '/log_train.txt'
        f = open(log_file)
        lines = f.readlines()
        f.close()
        eval_step = 0
        print_freq = 0
        batch_count = 0
        best_rank1 = 0
        btrk1_epoch = 0
        loss_list = []
        map_list = []
        cmc1_list = []
        cmc5_list = []
        cmc10_list = []
        cmc20_list = []
        tmap_list = []
        tcmc1_list = []
        tcmc5_list = []
        tcmc10_list = []
        tcmc20_list = []
        for line in lines:
            match = pattern_eval_step.search(line.strip())
            if match is not None:
                match_group_eval_step = match.groups()
                eval_step = int(match_group_eval_step[0])
                break
        for line in lines:
            match = pattern_print_freq.search(line.strip())
            if match is not None:
                match_group_print_freq = match.groups()
                print_freq = int(match_group_print_freq[0])
                break

        for line in lines:
            match = pattern_batch_count.search(line.strip())
            if match is not None:
                match_group_batch_count = match.groups()
                batch_count = int(match_group_batch_count[0])
                break

        for line in lines:
            match = pattern_best_rank1.search(line.strip())
            if match is not None:
                match_group_best_rank1 = match.groups()
                best_rank1 = float(match_group_best_rank1[0])
                break

        for line in lines:
            match = pattern_btrk1_epoch.search(line.strip())
            if match is not None:
                match_group_btrk1_epoch = match.groups()
                btrk1_epoch = int(match_group_btrk1_epoch[0])
                break

        for line in lines:
            match_loss = pattern_loss.search(line.strip())
            match_map = pattern_map.search(line.strip())
            match_cmc1 = pattern_cmc1.search(line.strip())
            match_cmc5 = pattern_cmc5.search(line.strip())
            match_cmc10 = pattern_cmc10.search(line.strip())
            match_cmc20 = pattern_cmc20.search(line.strip())
            if match_loss is not None:
                match_group_loss = match_loss.groups()
                loss_v = float(match_group_loss[0])
                loss_ave = float(match_group_loss[1])
                loss_list.append([loss_v, loss_ave])
            if match_map is not None:
                map_list += [float(match_map.groups()[0])] * eval_step
            if match_cmc1 is not None:
                cmc1_list += [float(match_cmc1.groups()[0])] * eval_step
            if match_cmc5 is not None:
                cmc5_list += [float(match_cmc5.groups()[0])] * eval_step
            if match_cmc10 is not None:
                cmc10_list += [float(match_cmc10.groups()[0])] * eval_step
            if match_cmc20 is not None:
                cmc20_list += [float(match_cmc20.groups()[0])] * eval_step
        assert len(map_list) == len(cmc1_list) == len(cmc5_list) == len(cmc10_list) == len(cmc20_list)
        # assert best_rank1 == cmc1_list[btrk1_epoch-1]

        if version_add_evaluate_train_data:
            for line in lines:
                match_tmap = pattern_tmap.search(line.strip())
                match_tcmc1 = pattern_tcmc1.search(line.strip())
                match_tcmc5 = pattern_tcmc5.search(line.strip())
                match_tcmc10 = pattern_tcmc10.search(line.strip())
                match_tcmc20 = pattern_tcmc20.search(line.strip())
                if match_tmap is not None:
                    tmap_list += [float(match_tmap.groups()[0])] * eval_step
                if match_tcmc1 is not None:
                    tcmc1_list += [float(match_tcmc1.groups()[0])] * eval_step
                if match_tcmc5 is not None:
                    tcmc5_list += [float(match_tcmc5.groups()[0])] * eval_step
                if match_tcmc10 is not None:
                    tcmc10_list += [float(match_tcmc10.groups()[0])] * eval_step
                if match_tcmc20 is not None:
                    tcmc20_list += [float(match_tcmc20.groups()[0])] * eval_step
            assert len(tmap_list) == len(tcmc1_list) == len(tcmc5_list) == len(tcmc10_list) == len(tcmc20_list)

        loss_list = np.array(loss_list)
        step = batch_count // print_freq
        loss_list_step = loss_list[step-1::step, :]
        map_list = np.array(map_list)
        cmc1_list = np.array(cmc1_list)
        cmc5_list = np.array(cmc5_list)
        cmc10_list = np.array(cmc10_list)
        cmc20_list = np.array(cmc20_list)
        train_info_dict[param] = [loss_list_step, loss_list, map_list, cmc1_list,
                                  cmc5_list, cmc10_list, cmc20_list, [best_rank1, btrk1_epoch]]

        if version_add_evaluate_train_data:
            tmap_list = np.array(tmap_list)
            tcmc1_list = np.array(tcmc1_list)
            tcmc5_list = np.array(tcmc5_list)
            tcmc10_list = np.array(tcmc10_list)
            tcmc20_list = np.array(tcmc20_list)
            train_info_dict[param] = [loss_list_step, loss_list,
                                      map_list, cmc1_list, cmc5_list, cmc10_list, cmc20_list,
                                      tmap_list, tcmc1_list, tcmc5_list, tcmc10_list, tcmc20_list,
                                      [best_rank1, btrk1_epoch]]

    return train_info_dict


train_info_dict = get_train_info()
type_list = ['-lssp', '-ls', '-mp', '-c1', '-c5', '-c10', '-c20', '-tmp', '-tc1', '-tc5', '-tc10', '-tc20']
type_plot_index = [0, 2, 3, 7, 8]
# type_plot_index = [1]
# type_plot_index = [0]
color = ['r', 'g', 'b', 'y', 'c', 'k', 'tab:purple', 'm']
# color = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']
type_index2 = 0
title = "vggm1024v2pre"

fig = plt.figure()
ax0 = fig.add_subplot(111, title=title)
for i, param in enumerate(log_param_list):
    for type_index1 in type_plot_index:
        x_range = range(len(train_info_dict[param][type_index1]))
        if type_index1 == 0 or type_index1 == 1:
            ax0.plot(x_range, train_info_dict[param][type_index1][:, type_index2],
                     color[i], label=param + type_list[type_index1])
            continue
        if type_index1 == 2:
            last_value_str = '-{:.2f}-e{:d}-{:.2f}'.format(train_info_dict[param][type_index1][-1],
                                                           train_info_dict[param][-1][1],
                                                           train_info_dict[param][-1][0])
            ax0.plot(x_range, train_info_dict[param][type_index1] / 100 * 6, color[i],
                     label=param + type_list[type_index1] + last_value_str)
            continue
        last_value_str = '-{:.2f}'.format(train_info_dict[param][type_index1][-1])
        ax0.plot(x_range, train_info_dict[param][type_index1] / 100 * 6, color[i],
                 label=param + type_list[type_index1] + last_value_str)
# ax0.legend(loc=0)  # best
# ax0.legend(loc=1)  # upper right
# ax0.legend(loc=2)  # upper left
# ax0.legend(loc=3)  # lower left
# ax0.legend(loc=4)  # lower right
fig.savefig(os.path.join(log_dir, 'vggm1024v3pre_xent_xtl_xtlv2_gstev2-2'))
plt.show()


