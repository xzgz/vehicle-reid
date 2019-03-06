from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
from collections import defaultdict
import scipy.io
import numpy as np
import cv2
import os


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
def display_ranked_result(query_feature,   query_label,   quim_path,         query_cam,
                          gallery_feature, gallery_label, gaim_path,         gallery_cam,
                          model_name,      features_name, show_match_images, reserve_same_camera,
                          result_dir,
                          id_index=0,      grid_width=3,  query_index=0,     single_gallery_shot=False,
                          topk=2,          show_ground_true=False,           save_match_images=False):
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
        gallery_c = gallery_cam[gallery_index]
    else:
        gallery_f = gallery_feature
        gallery_l = gallery_label
        gaim_p = gaim_path
        gallery_c = gallery_cam

    query_index = query_index
    qf = query_feature[query_index]

    if model_name[:8] == 'VeRi-MGN':
        qf = query_feature[[query_index]]
        dist_array = cdist(qf, gallery_f)
        indices = np.argsort(dist_array, axis=1)[0]
    else:
        dist_array = np.dot(gallery_f, qf)
        indices = np.argsort(dist_array, axis=0)
        indices = indices[::-1]
    matches = (gallery_l[indices] == query_label[query_index])
    if reserve_same_camera:
        valid = [True]*len(matches)
    else:
        valid = (gallery_l[indices] != query_label[query_index]) | (gallery_c[indices] != query_cam[query_index])
    ranked_image_path = gaim_p[indices][valid]
    matches = matches[valid]
    correct_index = np.nonzero(matches)[0]
    print('query image path:')
    print(quim_path[query_index].strip())
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
    quim = cv2.imread(quim_path[query_index].strip())
    img_spatial_shape = quim.shape[:2]

    # generate match_images
    # quim = pad_image_func(img=quim, pad_color=pad_color3, pad_length=pad_length1)
    # quim = pad_image_func(img=quim, pad_color=pad_color3, pad_length=pad_length3)
    quim = pad_image_func(img=quim, pad_color=pad_color3,
                          pad_length=np.array(pad_length1)+np.array(pad_length3))
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

    # generate true_images
    ground_true_line = []
    ground_true = []
    gt_cnt = 0
    for i in correct_index:
        img = cv2.imread(ranked_image_path[i].strip())
        img = cv2.resize(img, (img_spatial_shape[1], img_spatial_shape[0]))
        img = pad_image_func(img=img, pad_color=pad_color3,
                             pad_length=np.array(pad_length1) + np.array(pad_length3))
        ground_true_line.append(img)
        gt_cnt += 1
        if gt_cnt % grid_width == 0:
            ground_true.append(np.concatenate(ground_true_line, axis=1))
            ground_true_line = []
    last_line_length = len(ground_true_line)
    pad_image = np.zeros((img_spatial_shape[0], img_spatial_shape[1], 3), dtype=quim.dtype)
    pad_image = pad_image_func(img=pad_image, pad_color=pad_color3,
                               pad_length=np.array(pad_length1) + np.array(pad_length3))
    if last_line_length > 0:
        if last_line_length < grid_width:
            pad_cnt = grid_width - len(ground_true_line)
            ground_true_line += [pad_image]*pad_cnt
        ground_true.append(np.concatenate(ground_true_line, axis=1))
    true_images = np.concatenate(ground_true, axis=0)

    if show_ground_true:
        plt.figure('true images')
        plt.imshow(true_images[:, :, (2, 1, 0)])
        plt.show()
    if show_match_images:
        plt.figure('matched images')
        plt.imshow(match_images[:, :, (2, 1, 0)])
        plt.show()
    if save_match_images:
        if single_gallery_shot:
            # image_name = model_name + '_' + features_name[:-4] + '_q' + str(query_index) \
            #              + '_gid' + str(id_index) + '_top' + str(topk) + '.jpg'
            # image_name_gt = model_name + '_' + features_name[:-4] + '_q' + str(query_index) \
            #              + '_gid' + str(id_index) + '_top' + str(topk) + 'T.jpg'
            image_name = 'vid' + str(query_label[query_index]) + '_q' + str(query_index) + '_gid' + str(id_index) \
                         + '_top' + str(topk) + '-' + model_name + '.jpg'
            image_name_gt = 'vid' + str(query_label[query_index]) + '_q' + str(query_index) + '_gid' + str(id_index) \
                            + '_top' + str(topk) + '-ZT-' + model_name + '.jpg'
        else:
            # image_name = model_name + '_' + features_name[:-4] + '_q' + str(query_index) + '_top' + str(topk) + '.jpg'
            # image_name_gt = model_name + '_' + features_name[:-4] + '_q' + str(query_index) + '_top' + str(topk) + 'T.jpg'
            image_name = 'vid' + str(query_label[query_index]) + '_q' + str(query_index) + '_top' + str(topk) \
                         + '-' + model_name + '.jpg'
            image_name_gt = 'vid' + str(query_label[query_index]) + '_q' + str(query_index) + '_top' + str(topk) \
                            + '-ZT-' + model_name + '.jpg'
        # spec_model_dir = os.path.join(model_dir, model_name)
        # image_dir = os.path.join(model_dir, 'VeRi-Matched-Image-SGS')
        image_dir = os.path.join(result_dir, 'VeRi-Matched-Image-RSC')
        # image_dir = os.path.join(model_dir, 'VeRi-Matched-Image')
        if not os.path.isdir(image_dir):
            os.makedirs(image_dir)
        cv2.imwrite(os.path.join(image_dir, image_name), match_images)
        cv2.imwrite(os.path.join(image_dir, image_name_gt), true_images)

def test_display_ranked_result():
    model_feature_name = [
                         ['VeRi-MGN-64', 'features_80.mat'],
                         ['VeRi-PCB-32', 'features_59.mat'],
                         ['VeRi-PCB-32-bk4', 'features_59.mat'],
                         ['VeRi-PCB-bk4-32', 'features_60.mat'],
                         ['VeRi-PCB-bk9-32', 'features_60.mat'],
                         ['VeRi-ResNet50-32', 'features_60.mat'],
                         ]
    for mfn in model_feature_name:
    # for mfn in [model_feature_name[-1]]:
        print('\n*************************************')
        print('Start visualize result of %s %s.' % (mfn[0], mfn[1]))
        model_name = mfn[0]
        features_name = mfn[1]
        spec_model_dir = os.path.join(model_dir, model_name)
        features_path = os.path.join(spec_model_dir, features_name)
        result = scipy.io.loadmat(features_path)
        query_feature = result['query_f']
        query_label = result['query_label'][0]
        query_cam = result['query_cam'][0]
        quim_path = result['quim_path']
        gallery_feature = result['gallery_f']
        gallery_label = result['gallery_label'][0]
        gaim_path = result['gaim_path']
        gallery_cam = result['gallery_cam'][0]
        result_dir = '/media/sdc/gysj/log-reid'

    query_index_list = [33, 48, 49, 123, 132, 1677, 1676, 1672, 1659,
                            1533, 1532, 1473, 1073, 1071, 485, 524]

        for query_index in query_index_list:
        # for query_index in [query_index_list[-1]]:
            topk = 47
            id_index = 6
            grid_width = 8

            # show_match_images = True
            show_match_images = False

            # show_ground_true = True
            show_ground_true = False

            # single_shot = True
            single_shot = False

            save_image = True
            # save_image = False

            reserve_same_camera = True
            # reserve_same_camera = False

            display_ranked_result(query_feature,   query_label,    quim_path, query_cam,
                                  gallery_feature, gallery_label,  gaim_path, gallery_cam,
                                  id_index=id_index,               grid_width=grid_width,
                                  query_index=query_index,         topk=topk,
                                  single_gallery_shot=single_shot, show_ground_true=show_ground_true,
                                  save_match_images=save_image,    model_name=model_name,
                                  features_name=features_name,     show_match_images=show_match_images,
                                  result_dir=result_dir,
                                  reserve_same_camera=reserve_same_camera)
test_display_ranked_result()



