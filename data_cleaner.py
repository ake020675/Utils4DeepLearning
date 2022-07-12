"""
用于深度学习数据清洗
    去除已损坏图片:
        无法用pil.Image打开的图片
    去除模糊图片
        用opencv提供的拉普拉斯算子求得清晰度，数值越小越模糊（通常以100位分界值）。
    去除相似图片
    https://blog.csdn.net/weixin_35132022/article/details/112514520

        对于一些通过视频抽帧得到的图片数据，连续图片相似度会很高，需要剔除相似度较高的图片数据。
        首先我们需要计算两张图片的相似度，计算相似度的方法通常有以下几种：

        通过直方图计算图片的相似度；
        通过哈希值，汉明距离计算；
        通过图片的余弦距离计算；
        通过图片的结构度量计算。
"""

import os
import shutil
from PIL import Image
from cv2 import cv2
import numpy as np


def is_image(file):
    isImage = False
    if file.endswith(('png', 'jpeg', 'jpg', 'bmp')):
        isImage = True
    return isImage


# 获取数据集存储大小、图片数量、破损图片数量
def get_data_info(dir_path):
    size = 0
    number = 0
    bad_number = 0
    for root, dirs, files in os.walk(dir_path):
        img_files = [file_name for file_name in files if is_image(file_name)]
        files_size = sum([os.path.getsize(os.path.join(root, file_name)) for file_name in img_files])
        files_number = len(img_files)
        size += files_size
        number += files_number
        for file in img_files:
            try:
                img = Image.open(os.path.join(root, file))
                img.load()
            except OSError:
                bad_number += 1
    return size / 1024 / 1024, number, bad_number


# 去除已损坏图片
def filter_bad(dir_path):
    filter_dir = os.path.join(os.path.dirname(dir_path), 'filter_bad')
    if not os.path.exists(filter_dir):
        os.mkdir(filter_dir)
    filter_number = 0
    for root, dirs, files in os.walk(dir_path):
        img_files = [file_name for file_name in files if is_image(file_name)]
        for file in img_files:
            file_path = os.path.join(root, file)
            try:
                Image.open(file_path).load()
            except OSError:
                shutil.move(file_path, filter_dir)
                filter_number += 1
    return filter_number


# 去除模糊图片
def filter_blurred(dir_path, thresh=100):
    filter_dir = os.path.join(os.path.dirname(dir_path), 'filter_blurred')
    if not os.path.exists(filter_dir):
        os.mkdir(filter_dir)
    filter_number = 0
    for root, dirs, files in os.walk(dir_path):
        img_files = [file_name for file_name in files if is_image(file_name)]
        for file in img_files:
            file_path = os.path.join(root, file)
            # img = cv2.imread(file_path)
            img = cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), -1)
            image_var = cv2.Laplacian(img, cv2.CV_64F).var()
            if image_var < thresh:
                shutil.move(file_path, filter_dir)
                filter_number += 1
    return filter_number


# 计算两张图片的相似度
def calc_similarity(img1_path, img2_path, sim_thresh=0.95):
    img1 = cv2.imdecode(np.fromfile(img1_path, dtype=np.uint8), -1)
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)
    H1 = cv2.calcHist([img1], [0], None, [256], [0, 256])  # 计算图直方图
    # H1 = cv2.calcHist([img1], [0, 1, 2],
    #                     None, [256, 256, 256], [0, 256, 0, 256, 0, 256])
    H1 = cv2.normalize(H1, H1, 0, 1, cv2.NORM_MINMAX, -1)  # 对图片进行归一化处理

    img2 = cv2.imdecode(np.fromfile(img2_path, dtype=np.uint8), -1)
    img2 = cv2.imdecode(np.fromfile(img2_path, dtype=np.uint8), -1)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2HSV)
    H2 = cv2.calcHist([img2], [0], None, [256], [0, 256])  # 计算图直方图
    # H2 = cv2.calcHist([img2], [0, 1, 2],
    #                   None, [256, 256, 256], [0, 256, 0, 256, 0, 256])
    H2 = cv2.normalize(H2, H2, 0, 1, cv2.NORM_MINMAX, -1)  # 对图片进行归一化处理
    similarity1 = cv2.compareHist(H1, H2, 0)  # 相似度比较
    # print('similarity:', similarity1)
    if similarity1 > sim_thresh:  # 0.98是阈值，可根据需求调整
        return True
    else:
        return False


# 去除相似度高的图片
def filter_similar(dir_path, step=20):
    filter_dir = os.path.join(os.path.dirname(dir_path), 'filter_similar')
    if not os.path.exists(filter_dir):
        os.mkdir(filter_dir)
    filter_number = 0
    for root, dirs, files in os.walk(dir_path):
        img_files = [file_name for file_name in files if is_image(file_name)]
        filter_list = []
        for index in range(len(img_files))[:-4]:
            if img_files[index] in filter_list:
                continue
            for idx in range(len(img_files))[(index + 1):(index + step)]:  # 连续step帧图片对比
                img1_path = os.path.join(root, img_files[index])
                img2_path = os.path.join(root, img_files[idx])
                if calc_similarity(img1_path, img2_path):
                    filter_list.append(img_files[idx])
                    filter_number += 1
        for item in filter_list:
            src_path = os.path.join(root, item)
            # shutil.move(src_path, filter_dir)
            if os.path.exists(src_path):
                save_name = os.path.basename(root)+'_'+item
                shutil.move(src_path, os.path.join(filter_dir, save_name))  # by zk

    return filter_number


def datacleaner(dataset_dir):

    data_info = get_data_info(dataset_dir)
    if data_info[2] > 0:
        filter_bad(dataset_dir)

    filter_blurred(dataset_dir)

    filter_similar(dataset_dir)
    # for cls in os.listdir(dataset_dir):
    #     cls_dir = os.path.join(dataset_dir, cls)


if __name__ == "__main__":
    # dataset_dir = r'C:\Users\zk\Desktop\NewAlgo_LineLaser\videos\validate_DL\ColorLine500\shading'
    dataset_dir = r'ColorLine300\real2'

    datacleaner(dataset_dir)
