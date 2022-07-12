# -*-coding:utf-8-*-
# split xml and images of multiple class to dir of single class

import shutil
import glob
import os
from xml.etree import ElementTree as ET
import numpy as np
import cv2


def split_single_class(dir_multi_class='test_20220510_024-039/', dir_single_class='single', class_names=None,
                       logPath=None):
    """
    将多类物体标注xml，拆分为单类物体标注xml，并保存目标框小图
    """
    # 统计不同样本类型的总数
    if logPath is None:
        logPath = os.path.join(dir_multi_class, 'log.txt')
    log_object = open(logPath, 'w+', encoding='utf-8')

    if class_names is None:
        class_names = list(range(1, 40))
    object_numbers = np.zeros_like(class_names)

    # src_impath = os.path.join(dir_multi_class, 'train2017')
    # src_impath = os.path.join(dir_multi_class, 'images')

    src_xmlpath = os.path.join(dir_multi_class, 'xml')

    xml_files = glob.glob(os.path.join(src_xmlpath, '*.xml'))
    for xml_file in xml_files:

        # lm 对xml文件进行解析
        tree = ET.parse(xml_file)  # 读取xml文件
        root = tree.getroot()  # 获取根节点
        file_name = root.find('filename').text  # image filename
        if '.' in file_name:
            file_name = file_name.split('.')[0]

        # 若xml中包含对应类别物体，则复制xml到目标文件夹
        objs = root.iter('object')
        for i, ob in enumerate(objs):

            # check cls name: should be a number
            cls_name = ob.find('name').text
            isnumber = cls_name.isdigit()
            if not isnumber:
                print('Error: {}: label name is not a number!'.format(file_name))
                log_object.write('Error: {}: label name is not a number!'.format(file_name))
                continue

            # cls_name should be in class_names
            if int(cls_name) not in class_names:
                continue

            cls_index = class_names.index(int(cls_name))  # cls_names starts with 1
            object_numbers[cls_index] += 1

            # 复制xml
            dst_xmlpath = os.path.join(dir_single_class, cls_name, 'xml')
            if not os.path.exists(dst_xmlpath):
                os.makedirs(dst_xmlpath)
            dst_xml = os.path.join(dst_xmlpath, file_name + '.xml')
            if not os.path.exists(dst_xml) or os.path.getsize(dst_xml) == 0:
                shutil.copy(xml_file, dst_xml)

            # # 复制原图
            # dst_impath = os.path.join(dir_single_class, cls_name, 'img')
            # if not os.path.exists(dst_impath):
            #     os.makedirs(dst_impath)
            # dst_im = os.path.join(dst_impath, file_name + '.png')
            # if not os.path.exists(dst_im):
            #     shutil.copy(os.path.join(src_impath, file_name + '.png'), dst_im)

    log_object.write('class_names: {}'.format(class_names) + '\n')
    log_object.write('object_numbers: {}'.format(object_numbers) + '\n')
    log_object.close()

    print('done!')


if __name__ == "__main__":
    dir_multi_class = r'test_20220510_024-039/'
    dir_single_class = dir_multi_class + 'single'
    # class_names = list(range(1, 40))
    class_names = [1, 2, 3]
    split_single_class(dir_multi_class, dir_single_class, class_names=class_names)

    # img_dirs = glob.glob(os.path.join(dir_single_class, '*/crop'))
    # for im_dir in img_dirs:
    #     # os.remove(im_dir)
    #     shutil.rmtree(im_dir)

    # dir_multi_class = r'F:\lm\Image11\data\VOCdevkit'
    # dir_single_class = r'F:\lm\Image11\single'
    # class_names = list(range(24, 47))
    # # class_names = ["%03d" % n for n in class_names]
    # split_single_class(dir_multi_class, dir_single_class, class_names=class_names)
