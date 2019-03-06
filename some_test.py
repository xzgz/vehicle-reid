import os
import re
import shutil
import numpy as np
from collections import defaultdict


def test1():
    name_test_path = '/home/gysj/pytorch-workspace/pytorch-study/data/VeRi/name_test.txt'
    name_train_path = '/home/gysj/pytorch-workspace/pytorch-study/data/VeRi/name_train.txt'
    name_query_path = '/home/gysj/pytorch-workspace/pytorch-study/data/VeRi/name_query.txt'
    image_test_path = '/home/gysj/pytorch-workspace/pytorch-study/data/VeRi/image_test_with_view_label'
    image_train_path = '/home/gysj/pytorch-workspace/pytorch-study/data/VeRi/image_train_with_view_label'
    # train_image_dir = os.path.join(os.path.dirname(name_train_path), 'train_images')
    print(os.path.dirname(image_train_path))
    print(os.path.basename(image_train_path))
    print(os.path.basename(name_test_path))
    # print(os.path.abspath(__file__))

    name_dir = os.path.dirname(name_test_path)
    name_test_with_view = os.path.join(name_dir, 'name_test_with_view.txt')
    name_train_with_view = os.path.join(name_dir, 'name_train_with_view.txt')
    name_query_with_view = os.path.join(name_dir, 'name_query_with_view.txt')

    def list_pictures(directory, ext='jpg|jpeg|bmp|png|ppm'):
        return sorted([f for root, _, files in os.walk(directory) for f in files
                       if re.match(r'([\w]+\.(?:' + ext + '))', f)])


    # name_with_view = name_test_with_view
    name_with_view = name_train_with_view
    # txt_path = name_test_path
    txt_path = name_train_path
    # img_dir = image_test_path
    img_dir = image_train_path

    f_test_txt = open(txt_path)
    f_test = f_test_txt.readlines()
    f_test_txt.close()

    pattern = re.compile('([\d]+)_c([\d]+)_([\d]+)_([\d]+)')
    pattern_with_view = re.compile('([\d]+)_([\d]+)_([\d]+)_([\d]+)_([\d]+)_([\d]+)')

    id_time_camid_dict = defaultdict(list)
    for pic_name in f_test:
        pic_name = pic_name.strip()
        match = pattern.search(pic_name)
        if match is None:
            print('{} cannot be matched.'.format(pic_name))
            continue
        id, camid, time1, time2 = match.groups()
        id_time_camid_dict[id + '_' + time1 + '_' + time2].append(['c' + camid, pic_name])

    repeated_id_time_cnt = 0
    id_time_camid_no_repeat = {}
    for k, v in id_time_camid_dict.items():
        id_time_camid_cnt = len(v)
        if id_time_camid_cnt != 1:
            print('id_time_camid_cnt:', id_time_camid_cnt, 'k:', k, 'v:', v)
            repeated_id_time_cnt += 1
        else:
            id_time_camid_no_repeat[k] = v[0]
    print('repeated_id_time_cnt: {}'.format(repeated_id_time_cnt))

    img_id = 0
    img_paths = list_pictures(img_dir)
    id_time_camid_no_repeat_with_view = {}
    img_id_camid_gopid_dict = defaultdict(list)
    for im_pth in img_paths:
        im_pth = im_pth.strip()
        match = pattern_with_view.search(im_pth)
        if match is None:
            print('{} cannot be matched.'.format(im_pth))
            continue
        gopid, _, _, id, time1, time2 = match.groups()
        key_id_time = id + '_' + time1 + '_' + time2
        if key_id_time in id_time_camid_no_repeat:
            val_id_time = id_time_camid_no_repeat[key_id_time]
            img_name_new = val_id_time[0] + '_' + im_pth
            id_time_camid_no_repeat_with_view[key_id_time] = [val_id_time[1], img_name_new]
            gopid, id = map(int, [gopid, id])
            im_pth = os.path.join(img_dir, im_pth)
            img_id_camid_gopid_dict[img_id].extend([id, 0, gopid, im_pth])
            img_id += 1
    f_with_view = open(name_with_view, 'w')
    for k, v in id_time_camid_no_repeat_with_view.items():
        f_with_view.write(v[0] + ' ' + v[1] + '\n')
    f_with_view.close()


    label = np.array(list(img_id_camid_gopid_dict.values()))
    lbl_id = label[:, 0].astype(np.int32)
    lbl_gopid = label[:, 2].astype(np.int32)
    id_set = set(lbl_id)
    gopid_set = set(lbl_gopid)
    id_cnt = {}
    gopid_cnt = {}
    for id in id_set:
        id_cnt[id] = len(np.where(lbl_id == id)[0])
    for gopid in gopid_set:
        gopid_cnt[gopid] = len(np.where(lbl_gopid == gopid)[0])

    # f_query_txt = open(name_query_path)
    # f_query = f_query_txt.readlines()
    # f_query_txt.close()
    # f_query_with_view = open(name_query_with_view, 'w')
    # query_missed_cnt = 0
    # for pic_name in f_query:
    #     pic_name = pic_name.strip()
    #     match = pattern.search(pic_name)
    #     if match is None:
    #         print('{} cannot be matched.'.format(pic_name))
    #         continue
    #     id, camid, time1, time2 = match.groups()
    #     key_id_time = id + '_' + time1 + '_' + time2
    #     if key_id_time in id_time_camid_no_repeat_with_view:
    #         assert pic_name == id_time_camid_no_repeat_with_view[key_id_time][0]
    #         f_query_with_view.write(pic_name + ' ' + id_time_camid_no_repeat_with_view[key_id_time][1] + '\n')
    #     else:
    #         print('In query set, image {} missed in labeling view process!'.format(pic_name))
    #         query_missed_cnt += 1
    # f_query_with_view.close()
    # print('query image count: {}'.format(len(f_query)))
    # print('Total missed {} images in query set!'.format(query_missed_cnt))
    # print('percent: {:.2%}'.format(query_missed_cnt / float(len(f_query))))


    print('There are {} images in original test set.'.format(len(f_test)))
    print('Use id, time1, time2 as key, the image count is {}.'.format(len(id_time_camid_dict)))
    print('There are {} images in test set with view labels'.format(len(img_paths)))
    print('There are {} images in test set with view labels and no repeated camera ids'
          .format(len(id_time_camid_no_repeat_with_view)))
    print('each vehicle image count:\n', list(id_cnt.values()), '\n', sum(id_cnt.values()))
    print('{} images have wrong in renaming process!'
          .format(len(img_id_camid_gopid_dict) - len(id_time_camid_no_repeat_with_view)))
    print('Total missed {} images in labeling view process!'.format(len(f_test) - len(id_time_camid_no_repeat_with_view)))
    print('percent: {:.2%}'.format((len(f_test) - len(id_time_camid_no_repeat_with_view)) / float(len(f_test))))
    print('vehicle count: {}'.format(len(id_cnt.values())))
    print('vehicle count under each view:', list(gopid_cnt.values()), '\n', sum(gopid_cnt.values()))
    print('view count: {}'.format(len(gopid_cnt)))
# test1()


def test2():
    file_name = '/home/weiying1/hyg/pytorch-workspace/pytorch-study/data/VeRi/name_query.txt'
    fn = open(file_name)
    im_nms = fn.readlines()
    fn.close()
    for im in im_nms:
        im_pth = os.path.join(os.path.dirname(file_name), 'image_test', im.strip())
        if not os.path.isfile(im_pth):
            print('No such file {}'.format(im_pth))
        else:
            os.remove(im_pth)
    # shutil.move('/home/weiying1/hyg/pytorch-workspace/pytorch-study/data/VeRi/11',
    #             '/home/weiying1/hyg/pytorch-workspace/pytorch-study/data/VeRi/jk_index.txt')
# test2()


def test3():
    pic_dir_train = '/media/sda1/sleep-data/gysj/pytorch-workspace-pytorch-study-data/VeRi/image_train'
    pic_dir_test = '/media/sda1/sleep-data/gysj/pytorch-workspace-pytorch-study-data/VeRi/image_test'
    def list_pictures(directory, ext='jpg|jpeg|bmp|png|ppm'):
        return sorted([f for root, _, files in os.walk(directory) for f in files
                       if re.match(r'([\w]+\.(?:' + ext + '))', f)])

    pattern = re.compile('([-\d]+)_c([\d]+)_([\d]+)')
    camera_time = defaultdict(list)
    time_image_name = defaultdict(list)

    image_name = list_pictures(pic_dir_train)
    for im_nm in image_name:
        match = pattern.search(im_nm)
        if match is not None:
            _, camid, time_num = map(int, match.groups())
            # if time_num not in camera_time[camid]:
            #     camera_time[camid].append(time_num)
            # else:
                # print('Image {} time repeat!'.format(im_nm))
            camera_time[camid].append(time_num)
            time_image_name[time_num].append(im_nm)
        else:
            print('{} is abnormal!'.format(im_nm))

    image_name = list_pictures(pic_dir_test)
    for im_nm in image_name:
        match = pattern.search(im_nm)
        if match is not None:
            _, camid, time_num = map(int, match.groups())
            # if time_num not in camera_time[camid]:
            #     camera_time[camid].append(time_num)
            # else:
            # print('Image {} time repeat!'.format(im_nm))
            camera_time[camid].append(time_num)
            time_image_name[time_num].append(im_nm)
        else:
            print('{} is abnormal!'.format(im_nm))

    print('Camera id: {}'.format(camera_time.keys()))
    time_image_name_key = sorted(time_image_name.keys())
    # print(time_image_name_key)
    print(time_image_name_key[:100])
    print(time_image_name_key[-100:])
    print(min(time_image_name_key), max(time_image_name_key), len(time_image_name_key))

    print('Each camera st && end time:')
    for camid in camera_time.keys():
        sorted_camera_time = sorted(camera_time[camid])
        print(sorted_camera_time[0], sorted_camera_time[-1])
        # print(sorted_camera_time)
    longer_2 = 0
    for tim in time_image_name_key:
        if len(time_image_name[tim]) > 2:
            # print(time_image_name[tim])
            longer_2 += 1
    print('Longer than 2 count {}, total count {}'.format(longer_2, len(time_image_name_key)))
# test3()


def test4():
    name_test_path = '/home/weiying1/hyg/pytorch-workspace/pytorch-study/data/VeRi/name_test.txt'
    name_train_path = '/home/weiying1/hyg/pytorch-workspace/pytorch-study/data/VeRi/name_train.txt'
    txt_path = name_test_path
    src_image_dir = os.path.join(os.path.dirname(txt_path), 'image_test')
    dst_image_dir = os.path.join(os.path.dirname(txt_path), 'image_test_re_arrange')
    new_origin_name = os.path.join(os.path.dirname(txt_path), 'name_test_new2origin.txt')
    # txt_path = name_train_path
    # src_image_dir = os.path.join(os.path.dirname(txt_path), 'image_train')
    # dst_image_dir = os.path.join(os.path.dirname(txt_path), 'image_train_re_arrange')
    # new_origin_name = os.path.join(os.path.dirname(txt_path), 'name_train_new2origin.txt')

    if not os.path.exists(dst_image_dir):
        os.makedirs(dst_image_dir)

    pattern = re.compile('([\d]+)_c([\d]+)_([\d]+)_([\d]+)')

    f_txt = open(txt_path)
    f = f_txt.readlines()
    f_txt.close()

    vehicle_name_dict = defaultdict(list)
    for img_name in f:
        img_name = img_name.strip()
        match = pattern.search(img_name)
        if match is not None:
            vehi_id, _, _, _ = match.groups()
        else:
            print('{} cannot be matched.'.format(img_name))
            continue
        img_pth = os.path.join(src_image_dir, img_name)
        vehicle_name_dict[vehi_id].append([img_name, img_pth])
    print('Total {} images, {} vehicles.'.format(len(f), len(vehicle_name_dict)))

    f_new = open(new_origin_name, 'w')
    vehicle_id_cnt = 1
    for k, v in vehicle_name_dict.items():
        vehicle_cnt = 1
        for img_name, img_pth in vehicle_name_dict[k]:
            f_new.write(str(vehicle_id_cnt) + ' ' + str(vehicle_cnt) + ' ' + img_name + '\n')
            dst_dir = os.path.join(dst_image_dir, str(vehicle_id_cnt))
            if not os.path.exists(dst_dir):
                os.makedirs(dst_dir)

            dst_img_pth = os.path.join(dst_dir, str(vehicle_cnt) + '.jpg')
            src_img_pth = img_pth
            shutil.copyfile(src_img_pth, dst_img_pth)
            vehicle_cnt += 1

        vehicle_id_cnt += 1
    f_new.close()
    print('Done.')
test4()


