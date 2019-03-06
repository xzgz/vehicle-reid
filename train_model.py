from __future__ import print_function
from __future__ import division

import time
from utils.avgmeter import AverageMeter
from utils.torchtools import set_bn_to_eval


def train(epoch, model, criterion, optimizer, trainloader, loss_type, print_freq, freeze_bn=False):
    losses = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    end = time.time()
    for batch_idx, image_labels in enumerate(trainloader):
        data_time.update(time.time() - end)

        imgs, pids, group_labels, _, _, _ = image_labels

        imgs, pids, group_labels = imgs.cuda(), pids.cuda(), group_labels.cuda()

        model.train()
        if freeze_bn:
            model.apply(set_bn_to_eval)
        outputs = model(imgs)

        if loss_type == 'xent':
            # loss_xent_p1 = criterion[0](outputs[0], pids)
            # loss_xent_p2 = criterion[1](outputs[1], pids)
            # loss = 0.5 * (loss_xent_p1 + loss_xent_p2)
            # loss = loss_xent_p1 + loss_xent_p2
            loss_xent = [criterion(embed_clfy, pids) for embed_clfy in outputs]
            loss_xent = sum(loss_xent) / len(loss_xent)
            loss = 2.0 * loss_xent
            # loss = sum(loss_xent) / len(loss_xent)

        elif loss_type in ['xent_triplet', 'xent_tripletv2', 'xent_triplet_sqrt', 'xent_triplet_squa']:
            loss = criterion(outputs[0], outputs[1], pids, group_labels)
        else:
            raise KeyError("Unsupported loss: {}".format(loss_type))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)

        losses.update(loss.item(), pids.size(0))

        if (batch_idx + 1) % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Train time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t'
                  'Load data time {data_time.val:.4f}s ({data_time.avg:.4f}s)\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                epoch + 1, batch_idx + 1, len(trainloader),
                batch_time=batch_time,
                data_time=data_time, loss=losses))

        end = time.time()



