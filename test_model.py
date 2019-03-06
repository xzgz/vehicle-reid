from __future__ import print_function
from __future__ import division

import time
import torch
import datetime
import scipy.io
import numpy as np

from utils.avgmeter import AverageMeter
from eval_lib.eval_metrics import evaluate
from eval_lib.evaluate_vehicleid import cmc_common_oneshot_v2, cmc_vehicleid


def test(model, queryloader, galleryloader, train_query_loader, train_gallery_loader, use_gpu, test_batch, loss_type,
         euclidean_distance_loss, epoch, use_metric_cuhk03=False, ranks=[1, 5, 10, 20], return_distmat=False):
    batch_time = AverageMeter()

    model.eval()

    with torch.no_grad():
        tqf, tq_pids, tq_camids = [], [], []
        for batch_idx, (imgs, pids, _, _, camids, _) in enumerate(train_query_loader):
            if use_gpu: imgs = imgs.cuda()

            end = time.time()
            features = model(imgs)
            batch_time.update(time.time() - end)
            features = features.data.cpu()

            tqf.append(features)
            tq_pids.extend(pids)
            tq_camids.extend(camids)
        tqf = torch.cat(tqf, 0)
        tq_pids = np.asarray(tq_pids)
        tq_camids = np.asarray(tq_camids)
        print("Extracted features for train_query set, obtained {}-by-{} matrix".format(tqf.size(0), tqf.size(1)))
        print("==> BatchTime(s)/BatchSize(img): {:.3f}/{}".format(batch_time.avg, test_batch))
        tgf, tg_pids, tg_camids = [], [], []
        for batch_idx, (imgs, pids, _, _, camids, _) in enumerate(train_gallery_loader):
            if use_gpu: imgs = imgs.cuda()

            end = time.time()
            features = model(imgs)
            batch_time.update(time.time() - end)
            features = features.data.cpu()

            tgf.append(features)
            tg_pids.extend(pids)
            tg_camids.extend(camids)
        tgf = torch.cat(tgf, 0)
        tg_pids = np.asarray(tg_pids)
        tg_camids = np.asarray(tg_camids)
        print("Extracted features for train_gallery set, obtained {}-by-{} matrix".format(tgf.size(0), tgf.size(1)))
        print("==> BatchTime(s)/BatchSize(img): {:.3f}/{}".format(batch_time.avg, test_batch))

    print("Start compute distmat.")
    if loss_type in euclidean_distance_loss:
        m, n = tqf.size(0), tgf.size(0)
        distmat = torch.pow(tqf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
                  torch.pow(tgf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
        distmat.addmm_(1, -2, tqf, tgf.t())
        distmat = distmat.numpy()
    elif loss_type == 'angle':
        tvec_dot = torch.matmul(tqf, tgf.t())
        tqf_len = tqf.norm(dim=1, keepdim=True)
        tgf_len = tgf.norm(dim=1, keepdim=True)
        tvec_len = torch.matmul(tqf_len, tgf_len.t()) + 1e-5
        distmat = -torch.div(tvec_dot, tvec_len).numpy()
    else:
        raise KeyError("Unsupported loss: {}".format(loss_type))
    print("Compute distmat done.")
    print("distmat shape:", distmat.shape)
    print("Start computing CMC and mAP")
    start_time = time.time()
    cmc, mAP = evaluate(distmat, tq_pids, tg_pids, tq_camids, tg_camids,
                        use_metric_cuhk03=use_metric_cuhk03, use_cython=False)
    elapsed = round(time.time() - start_time)
    elapsed = str(datetime.timedelta(seconds=elapsed))
    print("Evaluate train data time (h:m:s): {}.".format(elapsed))
    print("Train data results ----------")
    print("Epoch {} trmAP: {:.2%}".format(epoch, mAP))
    print("CMC curve")
    for r in ranks:
        print("Epoch {} trRank-{:<3}: {:.2%}".format(epoch, r, cmc[r - 1]))
    print("------------------")


    with torch.no_grad():
        qf, q_pids, q_camids, q_paths = [], [], [], []
        for batch_idx, (imgs, pids, _, _, camids, paths) in enumerate(queryloader):
            if use_gpu: imgs = imgs.cuda()

            end = time.time()
            features = model(imgs)
            batch_time.update(time.time() - end)
            features = features.data.cpu()

            qf.append(features)
            q_pids.extend(pids)
            q_camids.extend(camids)
            q_paths.extend(paths)
        qf = torch.cat(qf, 0)
        q_pids = np.asarray(q_pids)
        q_camids = np.asarray(q_camids)
        q_paths = np.asarray(q_paths)
        print("Extracted features for query set, obtained {}-by-{} matrix".format(qf.size(0), qf.size(1)))
        print("==> BatchTime(s)/BatchSize(img): {:.3f}/{}".format(batch_time.avg, test_batch))
        gf, g_pids, g_camids, g_paths = [], [], [], []
        for batch_idx, (imgs, pids, _, _, camids, paths) in enumerate(galleryloader):
            if use_gpu: imgs = imgs.cuda()

            end = time.time()
            features = model(imgs)
            batch_time.update(time.time() - end)
            features = features.data.cpu()

            gf.append(features)
            g_pids.extend(pids)
            g_camids.extend(camids)
            g_paths.extend(paths)
        gf = torch.cat(gf, 0)
        g_pids = np.asarray(g_pids)
        g_camids = np.asarray(g_camids)
        g_paths = np.asarray(g_paths)
        print("Extracted features for gallery set, obtained {}-by-{} matrix".format(gf.size(0), gf.size(1)))
        print("==> BatchTime(s)/BatchSize(img): {:.3f}/{}".format(batch_time.avg, test_batch))

    print("Start compute distmat.")
    if loss_type in euclidean_distance_loss:
        m, n = qf.size(0), gf.size(0)
        distmat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
                  torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
        distmat.addmm_(1, -2, qf, gf.t())
        distmat = distmat.numpy()
    elif loss_type == 'angle':
        vec_dot = torch.matmul(qf, gf.t())
        qf_len = qf.norm(dim=1, keepdim=True)
        gf_len = gf.norm(dim=1, keepdim=True)
        vec_len = torch.matmul(qf_len, gf_len.t()) + 1e-5
        distmat = -torch.div(vec_dot, vec_len).numpy()
    else:
        raise KeyError("Unsupported loss: {}".format(loss_type))
    print("Compute distmat done.")
    print("distmat shape:", distmat.shape)
    # result = {'query_f': qf.numpy(),
    #           'query_cam': q_camids, 'query_label': q_pids, 'quim_path': q_paths,
    #           'gallery_f': gf.numpy(),
    #           'gallery_cam': g_camids, 'gallery_label': g_pids, 'gaim_path': g_paths}
    # scipy.io.savemat(os.path.join(args.save_dir, 'features_' + str(60) + '.mat'), result)
    # dist_mat_dict = {'dist_mat': distmat}
    # scipy.io.savemat(os.path.join(args.save_dir, 'features_' + str(60) + '_dist.mat'), dist_mat_dict)
    print("Start computing CMC and mAP")
    start_time = time.time()
    cmc, mAP = evaluate(distmat, q_pids, g_pids, q_camids, g_camids,
                        use_metric_cuhk03=use_metric_cuhk03, use_cython=False)
    elapsed = round(time.time() - start_time)
    elapsed = str(datetime.timedelta(seconds=elapsed))
    print("Evaluate test data time (h:m:s): {}.".format(elapsed))
    print("Test data results ----------")
    print("Epoch {} temAP: {:.2%}".format(epoch, mAP))
    print("CMC curve")
    for r in ranks:
        print("Epoch {} teRank-{:<3}: {:.2%}".format(epoch, r, cmc[r - 1]))
    print("------------------")

    if return_distmat:
        return distmat
    return cmc[0], mAP


def test_vehicleid(model, queryloader, galleryloader, train_query_loader, train_gallery_loader, use_gpu, test_batch,
                   loss_type, euclidean_distance_loss, epoch, use_metric_cuhk03=False, ranks=[1, 5, 10, 20],
                   return_distmat=False):
    batch_time = AverageMeter()

    model.eval()

    with torch.no_grad():
        qf, q_pids, q_paths = [], [], []
        for batch_idx, (imgs, pids, _, _, _, paths) in enumerate(queryloader):
            if use_gpu: imgs = imgs.cuda()

            end = time.time()
            features = model(imgs)
            batch_time.update(time.time() - end)
            features = features.data.cpu()

            qf.append(features)
            q_pids.extend(pids)
            q_paths.extend(paths)
        qf = torch.cat(qf, 0)
        q_pids = np.asarray(q_pids)
        q_paths = np.asarray(q_paths)
        print("Extracted features for query set, obtained {}-by-{} matrix".format(qf.size(0), qf.size(1)))
        print("==> BatchTime(s)/BatchSize(img): {:.3f}/{}".format(batch_time.avg, test_batch))
        gf, g_pids, g_paths = [], [], []
        for batch_idx, (imgs, pids, _, _, camids, paths) in enumerate(galleryloader):
            if use_gpu: imgs = imgs.cuda()

            end = time.time()
            features = model(imgs)
            batch_time.update(time.time() - end)
            features = features.data.cpu()

            gf.append(features)
            g_pids.extend(pids)
            g_paths.extend(paths)
        gf = torch.cat(gf, 0)
        g_pids = np.asarray(g_pids)
        g_paths = np.asarray(g_paths)
        print("Extracted features for gallery set, obtained {}-by-{} matrix".format(gf.size(0), gf.size(1)))
        print("==> BatchTime(s)/BatchSize(img): {:.3f}/{}".format(batch_time.avg, test_batch))

    # result = {'query_f': qf.numpy(),
    #           'query_cam': q_camids, 'query_label': q_pids, 'quim_path': q_paths,
    #           'gallery_f': gf.numpy(),
    #           'gallery_cam': g_camids, 'gallery_label': g_pids, 'gaim_path': g_paths}
    # scipy.io.savemat(os.path.join(args.save_dir, 'features_' + str(60) + '.mat'), result)
    # dist_mat_dict = {'dist_mat': distmat}
    # scipy.io.savemat(os.path.join(args.save_dir, 'features_' + str(60) + '_dist.mat'), dist_mat_dict)
    print("Start computing CMC and mAP")
    start_time = time.time()
    cmc, mAP = cmc_common_oneshot_v2(qf.numpy(), q_pids, gf.numpy(), g_pids, repeat=1, topk=50)
    elapsed = round(time.time() - start_time)
    elapsed = str(datetime.timedelta(seconds=elapsed))
    print("Evaluate test data time (h:m:s): {}.".format(elapsed))
    print("Test data results ----------")
    print("Epoch {} temAP: {:.2%}".format(epoch, mAP))
    print("CMC curve")
    for r in ranks:
        print("Epoch {} teRank-{:<3}: {:.2%}".format(epoch, r, cmc[r - 1]))
    print("------------------")

    return cmc[0], mAP


def test_vehicleid_formal(model, probeloader, galleryloader, train_query_loader, train_gallery_loader, use_gpu,
                          test_batch, loss_type, euclidean_distance_loss, epoch, use_metric_cuhk03=False,
                          ranks=[1, 5, 10, 20], return_distmat=False):
    batch_time = AverageMeter()

    model.eval()

    with torch.no_grad():
        pf, p_pids, p_paths = [], [], []
        for batch_idx, (imgs, pids, _, _, _, paths) in enumerate(probeloader):
            if use_gpu: imgs = imgs.cuda()

            end = time.time()
            features = model(imgs)
            batch_time.update(time.time() - end)
            features = features.data.cpu()

            pf.append(features)
            p_pids.extend(pids)
            p_paths.extend(paths)
        pf = torch.cat(pf, 0)
        p_pids = np.asarray(p_pids)
        p_paths = np.asarray(p_paths)
        print("Extracted features for query set, obtained {}-by-{} matrix".format(pf.size(0), pf.size(1)))
        print("==> BatchTime(s)/BatchSize(img): {:.3f}/{}".format(batch_time.avg, test_batch))

    # result = {'query_f': qf.numpy(),
    #           'query_cam': q_camids, 'query_label': q_pids, 'quim_path': q_paths,
    #           'gallery_f': gf.numpy(),
    #           'gallery_cam': g_camids, 'gallery_label': g_pids, 'gaim_path': g_paths}
    # scipy.io.savemat(os.path.join(args.save_dir, 'features_' + str(60) + '.mat'), result)
    # dist_mat_dict = {'dist_mat': distmat}
    # scipy.io.savemat(os.path.join(args.save_dir, 'features_' + str(60) + '_dist.mat'), dist_mat_dict)
    print("Start computing CMC and mAP")
    start_time = time.time()
    cmc, mAP = cmc_vehicleid(pf.numpy(), p_pids, repeat=10, topk=50)
    elapsed = round(time.time() - start_time)
    elapsed = str(datetime.timedelta(seconds=elapsed))
    print("Evaluate test data time (h:m:s): {}.".format(elapsed))
    print("Test data results ----------")
    print("Epoch {} temAP: {:.2%}".format(epoch, mAP))
    print("CMC curve")
    for r in ranks:
        print("Epoch {} teRank-{:<3}: {:.2%}".format(epoch, r, cmc[r - 1]))
    print("------------------")

    return cmc[0], mAP


