import scipy.io
import numpy as np
from collections import defaultdict
import cv2
import matplotlib.pyplot as plt
# sys.path.insert(0, '../open-reid/reid')
# from evaluation_metrics.ranking import cmc, mean_ap


# name = 'PCB-32'
# name = 'VehicleID-ResNet50-32'
# name = 'VehicleID-ResNet50-64'
# name = 'VehicleID-PCB-32'
# name = 'VehicleID-DenseNet121-32'
name = 'VehicleID-DenseNet121-56'
# name = 'VeRi-PCB-32-bk4'
# name = 'VeRi-PCB-32-bk9'
features_name = 'features_60.mat'
# features_name = 'features_probe_60.mat'


def _unique_sample(ids_dict, num):
    mask = np.zeros(num, dtype=np.bool)
    for _, indices in ids_dict.items():
        i = np.random.choice(indices)
        mask[i] = True
    return mask


def compute_dist_mat(query_feature, gallery_feature, is_save=False, save_path=None):
    qim_cnt = len(query_feature)
    gim_cnt = len(gallery_feature)
    dist_mat = np.zeros((qim_cnt, gim_cnt), dtype='float32')
    for i in range(qim_cnt):
        dist_mat[i] = np.dot(gallery_feature, query_feature[i])
        # print(i)
    if is_save:
        dist_mat_dict = {'dist_mat': dist_mat}
        scipy.io.savemat(save_path, dist_mat_dict)
    else:
        return dist_mat


def cmc_common(query_feature, query_label, gallery_feature, gallery_label, dist_mat=None,
               single_gallery_shot=False, repeat=10, topk=None):
    quim_cnt = len(query_feature)
    gaim_cnt = len(gallery_feature)
    if dist_mat is None:
        dist_mat = compute_dist_mat(query_feature, gallery_feature)

    indices = np.argsort(dist_mat, axis=1)
    indices = indices[:, ::-1]
    matches = (gallery_label[indices] == query_label[:, np.newaxis])
    if topk is None:
        cmc = np.zeros(gaim_cnt)
    else:
        cmc = np.zeros(topk)
    num_valid_queries = 0

    for i in range(quim_cnt):
        if not np.any(matches[i, :]):
            continue
        if single_gallery_shot:
            repeat = repeat
            gallery_id = gallery_label[indices[i]]
            gallery_id_dict = defaultdict(list)
            for j, x in enumerate(gallery_id):
                gallery_id_dict[x].append(j)
        else:
            repeat = 1
        for _ in range(repeat):
            if single_gallery_shot:
                sampled = _unique_sample(gallery_id_dict, gaim_cnt)
                index = np.nonzero(matches[i, sampled])[0]
            else:
                index = np.nonzero(matches[i, :])[0]
            if topk is None:
                cmc[index[0]:] += 1
            else:
                if index[0] < topk:
                    cmc[index[0]:] += 1
            num_valid_queries += 1

        if num_valid_queries == 0:
            raise RuntimeError("No valid query")
    print(num_valid_queries)
    return cmc/num_valid_queries


'''
Similar to cmc_vehicleid, the difference is that the query image set is fixed and gallery image set is randomly choose
from a prepared gallery image list.
'''
def cmc_common_oneshot_v2(query_feature, query_label, gallery_feature, gallery_label, repeat=10, topk=50):
    quim_cnt = len(query_feature)
    id_dict = defaultdict(list)
    for index, key in enumerate(gallery_label):
        id_dict[key].append(index)
    if topk is None:
        cmc = np.zeros(len(id_dict))
    else:
        cmc = np.zeros(topk)
    num_valid_queries = 0

    ave_cmc = []
    ave_mAP = []
    for _ in range(repeat):
        # num_valid_queries = 0
        # cmc = np.zeros(len(id_dict))
        gallery_index = []
        for key, index_list in id_dict.items():
            # i = np.random.choice(index_list)
            i = index_list[0]
            gallery_index.append(i)
        gallery_f = gallery_feature[gallery_index]
        gallery_l = gallery_label[gallery_index]
        dist_mat = compute_dist_mat(query_feature, gallery_f)
        indices = np.argsort(dist_mat, axis=1)
        indices = indices[:, ::-1]

        matches = (gallery_l[indices] == query_label[:, np.newaxis]).astype(np.int32)

        all_cmc = []
        all_AP = []
        num_valid_q = 0.0   # number of valid query
        for i in range(quim_cnt):
            orig_cmc = matches[i]

            if not np.any(orig_cmc):
                # this condition is true when query identity does not appear in gallery
                continue

            cmc = orig_cmc.cumsum()
            cmc[cmc > 1] = 1

            all_cmc.append(cmc[:topk])
            num_valid_q += 1.0

            num_rel = orig_cmc.sum()
            AP = 0
            matched_count = 0
            tmp_cmc = orig_cmc.cumsum().astype(np.float32)

            if orig_cmc[0]:
                AP += 1.0 / num_rel
                matched_count += 1
            for i in range(1, len(orig_cmc)):
                if orig_cmc[i]:
                    AP += ((tmp_cmc[i] - tmp_cmc[i - 1]) / num_rel) \
                          * ((tmp_cmc[i] / (i + 1.) + tmp_cmc[i - 1] / i) / 2)
                    matched_count += 1
                if matched_count == num_rel:
                    break

            all_AP.append(AP)

        assert num_valid_q > 0, "Error: all query identities do not appear in gallery"
        all_cmc = np.asarray(all_cmc).astype(np.float32)
        all_cmc = all_cmc.sum(0) / num_valid_q
        mAP = np.mean(all_AP)
        ave_cmc.append(all_cmc)
        ave_mAP.append(mAP)
        print(_, 'Rank1-5:', all_cmc[:5], 'mAP:', mAP)

    ave_cmc = np.asarray(ave_cmc).sum(0) / repeat
    ave_mAP = np.mean(ave_mAP)

    return ave_cmc, ave_mAP


    #     matches = (gallery_l[indices] == query_label[:, np.newaxis])
    #
    #     for i in range(quim_cnt):
    #         if not np.any(matches[i, :]):
    #             continue
    #         index = np.nonzero(matches[i, :])[0]
    #         if topk is None:
    #             cmc[index[0]:] += 1
    #         else:
    #             if index[0] < topk:
    #                 cmc[index[0]:] += 1
    #         num_valid_queries += 1
    #
    #     # print(_, (cmc/num_valid_queries)[[0, 4, 9]])
    #     if num_valid_queries == 0:
    #         raise RuntimeError("No valid query")
    # # print(num_valid_queries)
    # return cmc / num_valid_queries, 0


'''
One image is randomly selected from one identity to obtain a gallery set with 2,400 images, then the remaining 17,638
images are all used as probe images.
'''
def cmc_vehicleid(probe_feature, probe_label, repeat=10, topk=None):
    id_dict = defaultdict(list)
    for index, key in enumerate(probe_label):
        id_dict[key].append(index)
    if topk is None:
        cmc = np.zeros(len(id_dict))
    else:
        cmc = np.zeros(topk)
    # all_index = np.array(list(range(len(probe_label))))
    all_index = np.arange(len(probe_label))
    num_valid_queries = 0

    ave_cmc = []
    ave_mAP = []
    for _ in range(repeat):
        gallery_index = []
        for key, index_list in id_dict.items():
            i = np.random.choice(index_list)
            gallery_index.append(i)
        query_index = all_index[np.in1d(all_index, gallery_index, invert=True)]

        quim_cnt = len(query_index)
        gallery_feature = probe_feature[gallery_index]
        query_feature = probe_feature[query_index]
        # for i in range(quim_cnt):
        #     # dist_mat[i] = np.dot(probe_feature[gallery_index], probe_feature[query_index][i])
        #     dist_mat[i] = np.dot(gallery_feature, query_feature[i])
        #     # print(i)
        dist_mat = compute_dist_mat(query_feature, gallery_feature)
        indices = np.argsort(dist_mat, axis=1)
        indices = indices[:, ::-1]

        matches = (probe_label[gallery_index][indices] == probe_label[query_index][:, np.newaxis]).astype(np.int32)

        all_cmc = []
        all_AP = []
        num_valid_q = 0.0  # number of valid query
        for i in range(quim_cnt):
            orig_cmc = matches[i]

            if not np.any(orig_cmc):
                # this condition is true when query identity does not appear in gallery
                continue

            cmc = orig_cmc.cumsum()
            cmc[cmc > 1] = 1

            all_cmc.append(cmc[:topk])
            num_valid_q += 1.0

            num_rel = orig_cmc.sum()
            AP = 0
            matched_count = 0
            tmp_cmc = orig_cmc.cumsum().astype(np.float32)

            if orig_cmc[0]:
                AP += 1.0 / num_rel
                matched_count += 1
            for i in range(1, len(orig_cmc)):
                if orig_cmc[i]:
                    AP += ((tmp_cmc[i] - tmp_cmc[i - 1]) / num_rel) \
                          * ((tmp_cmc[i] / (i + 1.) + tmp_cmc[i - 1] / i) / 2)
                    matched_count += 1
                if matched_count == num_rel:
                    break

            all_AP.append(AP)

        assert num_valid_q > 0, "Error: all query identities do not appear in gallery"
        all_cmc = np.asarray(all_cmc).astype(np.float32)
        all_cmc = all_cmc.sum(0) / num_valid_q
        mAP = np.mean(all_AP)
        ave_cmc.append(all_cmc)
        ave_mAP.append(mAP)
        print(_, 'Rank1-5:', all_cmc[:5], 'mAP:', mAP)

    ave_cmc = np.asarray(ave_cmc).sum(0) / repeat
    ave_mAP = np.mean(ave_mAP)

    return ave_cmc, ave_mAP


    #     matches = (probe_label[gallery_index][indices] == probe_label[query_index][:, np.newaxis])
    #     for i in range(quim_cnt):
    #         if not np.any(matches[i, :]):
    #             continue
    #         index = np.nonzero(matches[i, :])[0]
    #         if topk is None:
    #             cmc[index[0]:] += 1
    #         else:
    #             if index[0] < topk:
    #                 cmc[index[0]:] += 1
    #         num_valid_queries += 1
    #     if num_valid_queries == 0:
    #         raise RuntimeError("No valid query")
    #     print(_, (cmc/num_valid_queries)[[0, 4, 9]])
    # print(num_valid_queries)
    # return cmc/num_valid_queries


def pad_image_func(img, pad_color, pad_length):
    img = img.transpose(2, 0, 1)
    img_b = img[0]
    img_g = img[1]
    img_r = img[2]
    img_b = np.pad(img_b, [[pad_length[0], pad_length[0]], [pad_length[1], pad_length[1]]], mode='constant',
                   constant_values=[[pad_color[2], pad_color[2]], [pad_color[2], pad_color[2]]])
    img_g = np.pad(img_g, [[pad_length[0], pad_length[0]], [pad_length[1], pad_length[1]]], mode='constant',
                   constant_values=[[pad_color[1], pad_color[1]], [pad_color[1], pad_color[1]]])
    img_r = np.pad(img_r, [[pad_length[0], pad_length[0]], [pad_length[1], pad_length[1]]], mode='constant',
                   constant_values=[[pad_color[0], pad_color[0]], [pad_color[0], pad_color[0]]])
    img = np.array([img_b, img_g, img_r])
    img = img.transpose(1, 2, 0)
    return img


def evaluate_vehicleid_with_cmc_common():
    result = scipy.io.loadmat('./model/'+name+'/'+features_name)
    query_feature = result['query_f']
    query_label = result['query_label'][0]
    gallery_feature = result['gallery_f']
    gallery_label = result['gallery_label'][0]
    # compute_dist_mat(query_feature, gallery_feature, is_save=True,
    #                  save_path='./model/'+name+'/'+features_name[:-4]+'_dist.mat')

    # # cmc_common_oneshot_v2 is almost equivalent to cmc_common
    # # when single_gallery_shot is True, but it's faster.
    # CMC = cmc_common_oneshot_v2(query_feature, query_label, gallery_feature, gallery_label, repeat=10)

    # dist_mat = scipy.io.loadmat('./model/'+name+'/'+features_name[:-4]+'_dist.mat')['dist_mat']
    CMC = cmc_common(query_feature, query_label, gallery_feature, gallery_label, dist_mat=None,
                     single_gallery_shot=False, repeat=10)
    print('Rank@1:%f Rank@5:%f Rank@10:%f' % (CMC[0], CMC[4], CMC[9]))
    fcmc = open('./model/'+name+'/'+features_name[:-4]+'_cmc.txt', 'w')
    for i in CMC:
        fcmc.write(str(i)+'\n')
    fcmc.close()
# evaluate_vehicleid_with_cmc_common()


def evaluate_vehicleid_with_cmc():
    names = ['VehicleID-ResNet50-32', 'VehicleID-ResNet50-64', 'VehicleID-PCB-32',
             'VehicleID-DenseNet121-32', 'VehicleID-DenseNet121-56']
    features_name = 'features_probe_60.mat'

    for name in names:
        result_probe = scipy.io.loadmat('./model/'+name+'/'+features_name)
        probe_feature = result_probe['probe_f']
        probe_label = result_probe['probe_label'][0]
        CMC = cmc_vehicleid(probe_feature, probe_label, repeat=100)
        print(name+': Rank@1:%f Rank@5:%f Rank@10:%f' % (CMC[0], CMC[4], CMC[9]))
        fcmc = open('./model/'+name+'/'+features_name[:-4]+'_cmc.txt', 'w')
        for i in CMC:
            fcmc.write(str(i)+'\n')
        fcmc.close()
# evaluate_vehicleid_with_cmc()


'''
Input: query_feature: Features of all query images extracted by the trained model, 2-D matrix, shape[0] denotes the
                         count of query images.
       query_label:   The corresponding ids of query images, 1-D vector.
       quim_path:     The corresponding absolute path of query images, 1-D vector, element type is numpy.str_, with 4
                      blank spaces in the end, which is necessary to be removed with strip function.
       gallery_feature: Similar to query_feature.
       gallery_label:   Similar to query_label.
       gaim_path:       Similar to quim_path.
       id_index:        When single_gallery_shot is true, choose a image of fixed index from each image list belong
                        to the same id.
       grid_width:      The count of images in every line When displaying the ranked gallery image list.
       query_index:     The index of image attempt to query in query image list.
       topk:            The number of images that display.
       single_gallery_shot: Whether to choose only one image from each id for gallery image list. If false, choose
                            all the images from each id for gallery image list.
       save_match_images:   Whether save the image grid of the ranked gallery image list.
'''
def display_ranked_result(query_feature, query_label, quim_path, gallery_feature, gallery_label,
                          gaim_path, id_index=0, grid_width=3, query_index=0, topk=2,
                          single_gallery_shot=False, show_ground_true=False, save_match_images=False):
    id_index = id_index
    if single_gallery_shot:
        id_dict = defaultdict(list)
        for index, key in enumerate(gallery_label):
            id_dict[key].append(index)
        gallery_index = []
        for key, index_list in id_dict.items():
            # i = np.random.choice(index_list)
            # gallery_index.append(i)
            if id_index >= len(index_list):
                gallery_index.append(index_list[-1])
            else:
                gallery_index.append(index_list[id_index])
        gallery_f = gallery_feature[gallery_index]
        gallery_l = gallery_label[gallery_index]
        gaim_p = gaim_path[gallery_index]
    else:
        gallery_f = gallery_feature
        gallery_l = gallery_label
        gaim_p = gaim_path

    query_index = query_index
    qf = query_feature[query_index]
    quim = cv2.imread(quim_path[query_index].strip())
    # print(type(str(quim_path[0])), str(quim_path[0]))
    # path = quim_path[0]
    # print(path[-5:], len(path))
    # print(path.strip()[-5:], len(path.strip()))
    # print(path)
    # print(path.strip())
    # print(type(path))

    dist_array = np.dot(gallery_f, qf)
    indices = np.argsort(dist_array, axis=0)
    indices = indices[::-1]
    ranked_image_path = gaim_p[indices]
    matches = (gallery_l[indices] == query_label[query_index])
    correct_index = np.nonzero(matches)[0]
    print('query image path:\n', quim_path[query_index].strip())
    print('correct_index:', correct_index)
    print('ground true count:', len(correct_index))
    print('query vehicle id:', query_label[query_index])

    grid_width = grid_width
    pad_color1 = [0, 255, 0]
    pad_color2 = [255, 0, 0]
    pad_color3 = [255, 255, 255]
    pad_length1 = [6, 6]
    pad_length2 = [6, 6]
    pad_length3 = [6, 6]
    match_result = []
    match_line = []
    img_spatial_shape = quim.shape[:2]
    quim = pad_image_func(img=quim, pad_color=pad_color3, pad_length=pad_length1)
    quim = pad_image_func(img=quim, pad_color=pad_color3, pad_length=pad_length3)
    match_result.append(quim)
    for i in range(topk):
        img = cv2.imread(ranked_image_path[i].strip())
        img = cv2.resize(img, (img_spatial_shape[1], img_spatial_shape[0]))
        if i in correct_index:
            img = pad_image_func(img=img, pad_color=pad_color1, pad_length=pad_length1)
        else:
            img = pad_image_func(img=img, pad_color=pad_color2, pad_length=pad_length2)
        img = pad_image_func(img=img, pad_color=pad_color3, pad_length=pad_length3)
        match_result.append(img)

        if (i+2) % grid_width == 0:
            match_line.append(np.concatenate(match_result, axis=1))
            match_result = []
    match_images = np.concatenate(match_line, axis=0)

    ground_true = []
    for i in correct_index:
        img = cv2.imread(ranked_image_path[i].strip())
        img = cv2.resize(img, (img_spatial_shape[1], img_spatial_shape[0]))
        ground_true.append(img)
    true_images = np.concatenate(ground_true, axis=1)

    if show_ground_true:
        plt.figure('true images')
        plt.imshow(true_images[:, :, (2, 1, 0)])
        plt.show()

    plt.figure('matched images')
    plt.imshow(match_images[:, :, (2, 1, 0)])
    plt.show()
    if save_match_images:
        image_dir = './model/' + name + '/'
        if single_gallery_shot:
            image_name = name + '_' + features_name[:-4] + '_q' + str(query_index) \
                         + '_gid' + str(id_index) + '_top' + str(topk) + '.jpg'
            image_name_gt = name + '_' + features_name[:-4] + '_q' + str(query_index) \
                         + '_gid' + str(id_index) + '_top' + str(topk) + 'T.jpg'
        else:
            image_name = name + '_' + features_name[:-4] + '_q' + str(query_index) + '_top' + str(topk) + '.jpg'
            image_name_gt = name + '_' + features_name[:-4] + '_q' + str(query_index) + '_top' + str(topk) + 'T.jpg'
        cv2.imwrite(image_dir+image_name, match_images)
        cv2.imwrite(image_dir+image_name_gt, true_images)


def test_display_ranked_result():
    result = scipy.io.loadmat('./model/'+name+'/'+features_name)
    query_feature = result['query_f']
    query_label = result['query_label'][0]
    quim_path = result['quim_path']
    gallery_feature = result['gallery_f']
    gallery_label = result['gallery_label'][0]
    gaim_path = result['gaim_path']

    # 296, 326, 356, 371, 372, 373, 374, 382
    # 396, 1034, 1102, 1179, 1181
    query_index = 1181
    topk = 15
    id_index = 4
    grid_width = 4
    single_shot = False
    show_true = True
    save_image = True
    display_ranked_result(query_feature, query_label, quim_path, gallery_feature, gallery_label, gaim_path,
                          id_index=id_index, grid_width=grid_width, query_index=query_index, topk=topk,
                          single_gallery_shot=single_shot, show_ground_true=show_true, save_match_images=save_image)
# test_display_ranked_result()


def write_correct_index():
    result = scipy.io.loadmat('./model/'+name+'/'+features_name)
    query_feature = result['query_f']
    query_label = result['query_label'][0]
    gallery_feature = result['gallery_f']
    gallery_label = result['gallery_label'][0]
    id_index = 4
    single_gallery_shot = False

    if single_gallery_shot:
        id_dict = defaultdict(list)
        for index, key in enumerate(gallery_label):
            id_dict[key].append(index)
        gallery_index = []
        for key, index_list in id_dict.items():
            # i = np.random.choice(index_list)
            # gallery_index.append(i)
            if id_index >= len(index_list):
                gallery_index.append(index_list[-1])
            else:
                gallery_index.append(index_list[id_index])
        gallery_f = gallery_feature[gallery_index]
        gallery_l = gallery_label[gallery_index]
    else:
        gallery_f = gallery_feature
        gallery_l = gallery_label

    quim_cnt = len(query_feature)
    f = open('./model/'+name+'/'+features_name[:-4]+'_gid'+str(id_index)+'_correct_index.txt', 'a')
    for i in range(quim_cnt):
        qf = query_feature[i]
        dist_array = np.dot(gallery_f, qf)
        indices = np.argsort(dist_array, axis=0)
        indices = indices[::-1]
        matches = (gallery_l[indices] == query_label[i])
        correct_index = np.nonzero(matches)[0]
        f.write(str(i)+': '+str(correct_index)+'\n')
        print(i)
    f.close()
# write_correct_index()


def draw_loss_error_curves():
    names = ['VehicleID-ResNet50-32', 'VehicleID-ResNet50-64', 'VehicleID-PCB-32',
             'VehicleID-DenseNet121-32', 'VehicleID-DenseNet121-56']
    yl_t = {}
    yl_v = {}
    ye_t = {}
    ye_v = {}
    x_epoch = {}

    for name in names:
        lef = open('./model/'+name+'/'+'loss_error.txt')
        le = lef.readlines()
        lef.close()
        le = [l.split() for l in le]
        le = [list(map(float, l)) for l in le]
        le = np.array(le)
        # print(le.shape, le.dtype)
        x_epoch[name] = list(range(len(le)))
        yl_t[name] = le[:, 0]
        yl_v[name] = le[:, 1]
        ye_t[name] = le[:, 2]
        ye_v[name] = le[:, 3]

    fig0 = plt.figure()
    fig1 = plt.figure()
    ax0 = fig0.add_subplot(111, title="loss")
    ax1 = fig1.add_subplot(111, title="error")
    # ignored_names = ['VehicleID-PCB-32']
    ignored_names = []
    show_names = names[0:1]

    for name in show_names:
        if name not in ignored_names:
            ax0.plot(x_epoch[name], yl_t[name], label='train_'+name)
            ax0.plot(x_epoch[name], yl_v[name], label='val_'+name)
            ax1.plot(x_epoch[name], ye_t[name], label='train_'+name)
            ax1.plot(x_epoch[name], ye_v[name], label='val_'+name)
    ax0.legend()
    ax1.legend()
    fig0.savefig('model/loss_curve.jpg')
    fig1.savefig('model/error_curve.jpg')
    plt.show()
# draw_loss_error_curves()


def draw_cmc_curves():
    names = ['VehicleID-ResNet50-32', 'VehicleID-ResNet50-64', 'VehicleID-PCB-32',
             'VehicleID-DenseNet121-32', 'VehicleID-DenseNet121-56']
    # cmc_filename = 'features_probe_60_cmc.txt'
    cmc_filename = 'features_60_cmc.txt'
    cmc_y = {}
    cmc_x = {}

    for name in names:
        cmcf = open('./model/'+name+'/'+cmc_filename)
        cmc = cmcf.readlines()
        cmcf.close()
        cmc = [c.strip() for c in cmc]
        cmc = list(map(float, cmc))
        cmc_y[name] = np.array(cmc)
        cmc_x[name] = np.arange(len(cmc))

    fig = plt.figure()
    ax0 = fig.add_subplot(111, title="CMC")
    ignored_names = []
    show_names = names[0:]
    # show_range = list(range(40))
    show_range = [0, 5, 10, 20, 30, 40, 50]
    for name in show_names:
        print(name+': Rank@1:%f Rank@5:%f Rank@10:%f' % (cmc_y[name][0], cmc_y[name][4], cmc_y[name][9]))
        if name not in ignored_names:
            ax0.plot(cmc_x[name][show_range], cmc_y[name][show_range], 'o-', label=name)
    ax0.legend()
    fig.savefig('model/cmc_curve.jpg')
    # plt.ylim(0)
    # vehicleid_state_of_the_art = plt.imread('./model/vehicleid_state_of_the_art.png')
    # plt.figure()
    # plt.imshow(vehicleid_state_of_the_art)
    plt.show()
# draw_cmc_curves()




