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

file_root = 'C:/Users/zk/Desktop/Project-411_classification/datasets/occlusion/'  # 当前文件夹下的所有图片
file_list = os.listdir(file_root)

save_path = r'C:/Users/zk/Desktop/Project-411_classification/datasets/occlusion_saved/'  # 保存目录
if not os.path.exists(save_path):
    os.mkdir(save_path)

i = 1

for img_name in file_list:

    if img_name.endswith('.png'):

        img_path = file_root + img_name

        xml_name = img_name.split('.')[0]
        xml_path = file_root + xml_name + '.xml'

        color_img = cv2.imread(img_path)

        tree = ET.ElementTree(file=xml_path)
        root = tree.getroot()

        boxes = []
        for elem in root.iter(tag='object'):
            # 获取矩形框坐标
            box = [int(el.text) for el in elem.find('bndbox')]
            strr_label = elem.find('name').text
            box_image = color_img[box[1]:box[3], box[0]:box[2]]
            if box[1] == box[3] or box[0] == box[2]:
                continue
            else:
                save_dir = os.path.join(save_path, strr_label)
                if not os.path.exists(save_dir):
                    os.mkdir(save_dir)

                save_name = os.path.join(save_dir,  xml_name + '_' + str(i) + '.png')
                cv2.imwrite(save_name, box_image)
            i = i + 1

        print(img_name)
