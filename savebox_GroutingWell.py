#!/usr/bin/python
# -*- coding: UTF-8 -*-

"""
made : liaoyuting
modified : zk, 2021.8.24
"""

import numpy as np
import xml.etree.ElementTree as ET
import cv2
import os
from cv2 import cv2
from PIL import Image

file_root = r'D:\火眼智能\注浆孔、灯\注浆孔数据集\augmentation\002大图_rotate15_标注\重影_错位'  # 当前文件夹下的所有图片
file_list = os.listdir(file_root)

save_path = file_root+'_小图'  # 保存目录
if not os.path.exists(save_path):
    os.makedirs(save_path)

i = 1

for img_name in file_list:

    if img_name.endswith('.png'):

        img_path = os.path.join(file_root, img_name)

        xml_name = img_name.split('.')[0]
        xml_path = os.path.join(file_root, xml_name + '.xml')

        if not os.path.exists(xml_path):
            continue

        # color_img = cv2.imread(img_path)
        # im = Image.open(img_path)
        image = np.array(Image.open(img_path))

        tree = ET.ElementTree(file=xml_path)
        root = tree.getroot()

        boxes = []
        for elem in root.iter(tag='object'):
            # 获取矩形框坐标
            box = [int(el.text) for el in elem.find('bndbox')]
            strr_label = elem.find('name').text
            box_image = image[box[1]:box[3], box[0]:box[2]]
            if box[1] == box[3] or box[0] == box[2]:
                continue
            else:
                save_dir = os.path.join(save_path, strr_label)
                if not os.path.exists(save_dir):
                    os.mkdir(save_dir)

                save_name = os.path.join(save_dir, xml_name + '_' + str(i) + '.png')
                # cv2.imwrite(save_name, box_image)
                Image.fromarray(box_image).save(save_name)

            i = i + 1

print('/n %d images processed' % (i-1))
