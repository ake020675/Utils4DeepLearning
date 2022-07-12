"""
    复制图像，创建分类数据集
"""

import os
import shutil
import glob
from cv2 import cv2
import numpy as np
from data_cleaner import datacleaner
from makeTxt_imagenet import get_map_dict


def brenner(img):
    """
    计算灰度图像清晰度
    :param img:narray 二维灰度图像
    :return: float 图像越清晰越大
    """
    shape = np.shape(img)
    out = 0
    for x in range(0, shape[0] - 2):
        for y in range(0, shape[1]):
            out += (int(img[x + 2, y]) - int(img[x, y])) ** 2
    return out / (shape[0] * shape[1])


def collectDataset_linelaser202203(src_dir=r'C:\Users\zk\Desktop\NewAlgo_LineLaser\videos\LineLaser202203',
                                   dirname='LaserLabel'):
    """
    从linelaser202203文件夹下，收集所有dirname子文件夹
    Args:
        src_dir:
        dirname:

    Returns:

    """
    dst_dir = os.path.join(src_dir, 'Dataset_' + dirname)
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)

    users = {'gd', 'zk', 'fyp', 'qjy'}
    # classes = {'shading', 'real2', 'real', 'print', 'laminate', 'laminate_laserline'}
    classes = {'real2', 'shading'}

    for user in users:
        user_dir = os.path.join(src_dir, user)
        for cls in classes:
            img_files = glob.glob(os.path.join(user_dir, cls, dirname + '/*.png'))
            save_dir = os.path.join(dst_dir, cls)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            for file in img_files:
                file_name = os.path.basename(file)
                save_path = os.path.join(save_dir, user + '_' + file_name)
                shutil.copy(file, save_path)


def collectDataset_linelaser(src_dir=r'C:\Users\zk\Desktop\NewAlgo_LineLaser\videos\LineLaser', dirname='LaserLabel'):
    """
    收集源路径下所有文件名为dirname的子文件夹
    Args:
        src_dir: 源数据路径
        dirname: 待收集子文件夹

    Returns:

    """
    dst_dir = os.path.join(src_dir, 'Dataset_' + dirname)
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)

    src_dirs = glob.glob(os.path.join(src_dir, '*', dirname))
    for dir in src_dirs:
        img_files = glob.glob(os.path.join(dir, '*.png'))
        cls = os.path.split(os.path.dirname(dir))[1]
        cls_dir = os.path.join(dst_dir, cls)
        if not os.path.exists(cls_dir):
            os.makedirs(cls_dir)

        for file in img_files:
            file_name = os.path.basename(file)
            save_path = os.path.join(cls_dir, file_name)
            shutil.copy(file, save_path)


# 统计非零面积
def getStats_NonZero(dir_path):
    bad_number = 0
    big_number = 0
    area_thresh = 400
    area_list = list()
    for file in os.listdir(dir_path):
        try:
            im = cv2.imread(os.path.join(dir_path, file), -1)
            gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            area = cv2.countNonZero(gray)
            # area = brenner(gray)

            area_list.append(area)
            # if area_thresh-10 < area < area_thresh+10:
            if area > area_thresh:
                big_number += 1
                cv2.namedWindow(file, 0)
                cv2.imshow(file, im)
                cv2.waitKey()
                cv2.destroyWindow(file)

        except OSError:
            bad_number += 1
    print('numbers of image:{} with area >:{}'
          .format(big_number, area_thresh))
    print('min area:{}, max area:{}, mean area:{}'
          .format(np.min(area_list), np.max(area_list), np.mean(area_list)))

    return bad_number


def count_number_per_label(label_list, dataset_dir):
    classes = os.listdir(dataset_dir)
    for cls in classes:
        if cls == 'meta':
            continue

        print('cls:{}'.format(cls))
        num_list = np.zeros(len(label_list))
        cls_dir = os.path.join(dataset_dir, cls)
        for file in os.listdir(cls_dir):
            labelcode = file[file.index('010'):file.index('010') + 10]
            label_index = label_list.index(labelcode)
            num_list[label_index] += 1
        print('label:{}'.format(label_list))
        print('sample numbers:{}'.format(num_list))


def makeTxt_trainval_laserlabel(root_dir, class_map_txt, label_list):
    """
    按标签号，为laserlabel数据集生成train/val/test列表
    Args:
        root_dir: 数据集根目录
        class_map_txt: 文件类别列表
        label_list: 所有标签号

    Returns:

    """
    num_label = len(label_list)
    train_labels = label_list[6:]
    # train_labels = label_list[:num_label-6]

    class_map_dict = get_map_dict(class_map_txt)
    class_dirs = os.listdir(root_dir)

    train_txt = os.path.join(root_dir, 'meta', "train.txt")
    val_txt = os.path.join(root_dir, 'meta', "val.txt")
    # test_txt = os.path.join(root_dir, 'meta', "test.txt")
    f_train = open(train_txt, "w")
    f_val = open(val_txt, "w")
    # f_test = open(test_txt, "w")

    for cls in class_dirs:
        if cls == 'meta':
            continue
        im_files = os.listdir(os.path.join(root_dir, cls))
        cls_label = class_map_dict[cls]

        for file in im_files:
            # get labelcode
            labelcode = file[file.index('010'):file.index('010') + 10]

            if labelcode in train_labels:
                f_train.write(cls + '/' + file + " " + cls_label + "\n")
            else:
                f_val.write(cls + '/' + file + " " + cls_label + "\n")
            # for file in test_files:
            #     f_test.write(cls + '/' + file + " " + cls_label + "\n")

    f_train.close()
    f_val.close()
    # f_test.close()


def getMask_ColorLine(blockRGB, sat_threshold=43, val_threshold=46):
    """
    by zk
    获取镭射区中有色区域（高亮高饱和）的掩膜图，有颜色的区域的值置为色调值，无颜色的区域的值置为0
    :param blockhsv:
    :param sat_threshold:
    :return:
    """

    # 自适应二值化：提取线条区域
    blockHSV = cv2.cvtColor(blockRGB, cv2.COLOR_BGR2HSV)
    _, sat, val = cv2.split(blockHSV)
    h, w = val.shape[:2]
    bin = cv2.adaptiveThreshold(val, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                cv2.THRESH_BINARY, 5, -5)
    side_length = 5
    bin[0:side_length, :] = 0
    bin[:, 0:side_length] = 0
    bin[:, w - side_length:w] = 0
    bin[h - side_length:h, :] = 0

    # 镭射区域高亮区域
    _, area_mask_V = cv2.threshold(val, val_threshold, 255, cv2.THRESH_BINARY)
    # 镭射区域高饱和区域
    _, area_mask_S = cv2.threshold(sat, sat_threshold, 255, cv2.THRESH_BINARY)
    Mask_Color = cv2.bitwise_and(area_mask_V, area_mask_S)

    # 彩色线条区域 = 线条区域 and 有色区域
    Mask_Color = cv2.bitwise_and(Mask_Color, bin)
    return bin, Mask_Color


# 筛除当前帧变色区域seg中太小的轮廓
def filterSeg(seg, min_area=6):
    res_seg = np.zeros_like(seg, dtype=seg.dtype)

    contours = list()
    if cv2.__version__.startswith('3'):
        _, contours, _ = cv2.findContours(seg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # opencv3
    if cv2.__version__.startswith('4'):
        contours, _ = cv2.findContours(seg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # opencv4

    for contour_index, contour in enumerate(contours):
        # 轮廓面积大于min_area 才纳入背景模型
        if cv2.contourArea(contour) >= min_area:
            cv2.drawContours(res_seg, contours, contour_index, 255, -1)
    return res_seg


def get_ColorLine_from_LaserLabel(
        src_dir=r'C:\Users\zk\Desktop\NewAlgo_LineLaser\videos\LineLaser202203'):
    """
    从所有防伪区域图像LaserLabel中提取ColorLine
    Args:
        src_dir:

    Returns:

    """
    src_dir = os.path.join(src_dir, 'Dataset_LaserLabel')
    if not os.path.exists(src_dir):
        print('Dataset_LaserLabel not exists!')
        return

    dst_dir = os.path.join(os.path.dirname(src_dir), 'ColorLine')
    if os.path.exists(dst_dir):
        shutil.rmtree(dst_dir)
    os.mkdir(dst_dir)

    stats_txt = os.path.join(dst_dir, 'stats.txt')
    f = open(stats_txt, 'w')

    class_list = os.listdir(src_dir)
    for cls in class_list:
        save_path = os.path.join(dst_dir, cls)
        os.mkdir(save_path)
        area_list = list()
        ratio_list = list()

        im_files = glob.glob(os.path.join(src_dir, cls + '/*.png'))
        if im_files is None:
            print('cls:{} not exists'.format(cls))
            continue

        print('cls:{}'.format(cls))
        f.write('cls:{}\n'.format(cls))

        for file in im_files:
            im = cv2.imread(file, 1)

            # mask_ColorLine: 彩色线条区域
            linemask, mask_ColorLine = getMask_ColorLine(im, 70, 70)
            mask_ColorLine = filterSeg(mask_ColorLine)

            area_color = cv2.countNonZero(mask_ColorLine)
            if area_color == 0:
                continue

            area_linemask = cv2.countNonZero(linemask)
            ratio = area_color / area_linemask if area_linemask > 0 else 0
            area_list.append(area_color)
            ratio_list.append(ratio)

            ColorLine = im.copy()
            ColorLine[mask_ColorLine == 0] = (0, 0, 0)
            save_name = os.path.join(save_path, os.path.basename(file))
            cv2.imwrite(save_name, ColorLine)

            # if ratio > 0.5:
            #     print('video:{} ratio:{}'.format(file, ratio))
            #     cv2.namedWindow('ColorLine', 0)
            #     cv2.imshow('ColorLine', ColorLine)
            #     cv2.namedWindow('linemask', 0)
            #     cv2.imshow('linemask', linemask)
            #     cv2.namedWindow('im', 0)
            #     cv2.imshow('im', im)
            #     cv2.waitKey()

        # print('min area:{}, max area:{}, mean area:{}'
        #       .format(np.min(area_list), np.max(area_list), np.mean(area_list)))
        # print('min ratio:{}, max ratio:{}, mean ratio:{}'
        #       .format(np.min(ratio_list), np.max(ratio_list), np.mean(ratio_list)))

        # f.write('min area:{}, max area:{}, mean area:{}\n'
        #         .format(np.min(area_list), np.max(area_list), np.mean(area_list)))
        # f.write('min ratio:{}, max ratio:{}, mean ratio:{}\n'
        #         .format(np.min(ratio_list), np.max(ratio_list), np.mean(ratio_list)))
    f.close()


def get_ColorLine_thresh(src_dir=r'C:\Users\zk\Desktop\NewAlgo_LineLaser\videos\LineLaser202203', area_thresh=300
                         ):
    """
    对所有ColorLine图像进行面积筛选，收集面积>阈值的ColorLine图像
    Args:
        src_dir:
        area_thresh:

    Returns:

    """
    src_dir = os.path.join(src_dir, 'ColorLine')
    if not os.path.exists(src_dir):
        print('Dir ColorLine not exists!')
        return

    class_names = [f for f in os.listdir(src_dir) if os.path.isdir(os.path.join(src_dir, f))]
    dst_dir = os.path.join(src_dir + str(area_thresh))
    if os.path.exists(dst_dir):
        shutil.rmtree(dst_dir)
    os.mkdir(dst_dir)

    for cls in class_names:
        cls_dir = os.path.join(src_dir, cls)
        save_dir = os.path.join(dst_dir, cls)
        os.mkdir(save_dir)

        for file in os.listdir(cls_dir):
            im = cv2.imread(os.path.join(cls_dir, file), -1)
            gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            area = cv2.countNonZero(gray)

            if area > area_thresh:
                shutil.copy(os.path.join(cls_dir, file), save_dir)


if __name__ == "__main__":
    # src_dir = r'C:\Users\zk\Desktop\NewAlgo_LineLaser\videos\Curved_LineLaser'
    # src_dir = r'C:\Users\zk\Desktop\NewAlgo_LineLaser\videos\LineLaser202204'

    # collectDataset_linelaser202203(src_dir)
    # get_ColorLine_from_LaserLabel(src_dir)
    # get_ColorLine_thresh(src_dir=src_dir, area_thresh=400)

    datacleaner(dataset_dir='ColorLine400_202204')

    # getStats_NonZero(dir_path=r'C:\Users\zk\Desktop\NewAlgo_LineLaser\videos\LineLaser202203\ColorLine500\HardSamples')

    # # make trainval Txt for dataset ColorLine500
    # # seed_list = os.listdir(r'C:\Users\zk\Desktop\NewAlgo_LineLaser\videos\LineLaser202203\zk\real2\seed')
    # label_list = ['0105168573', '0105168575', '0105168576', '0105168577', '0105168578', '0105168581', '0105168657',
    #               '0105168658', '0105168659', '0105168660', '0105168663', '0105168664', '0105168665', '0105168666',
    #               '0105168669', '0105168670', '0105168671', '0105168672', '0105168675', '0105168676', '0105168677']
    #
    # dataset_dir = 'ColorLine400_202203/'
    # # count_number_per_label(label_list, dataset_dir)
    #
    # classmap = os.path.join(dataset_dir, 'meta/classmap.txt')
    # makeTxt_trainval_laserlabel(dataset_dir, classmap, label_list)
