"""
by zk
20211115
对螺钉数据集图片进行分析
    大小、清晰度
输入：
输出：
"""

import os
import cv2
import numpy as np
from PIL import Image
from skimage import metrics, measure
import shutil


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


# 输入灰度图
# 输出NRSS清晰度
def NRSS(image):
    def sobel(img):
        x = cv2.Sobel(img, cv2.CV_16S, 1, 0)
        y = cv2.Sobel(img, cv2.CV_16S, 0, 1)

        absX = cv2.convertScaleAbs(x)  # 转回uint8
        absY = cv2.convertScaleAbs(y)

        dst = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)

        return dst

    def getBlock(G, Gr):
        (h, w) = G.shape
        G_blk_list = []
        Gr_blk_list = []
        sp = 6
        for i in range(sp):
            for j in range(sp):
                G_blk = G[int((i / sp) * h):int(((i + 1) / sp) * h), int((j / sp) * w):int(((j + 1) / sp) * w)]
                Gr_blk = Gr[int((i / sp) * h):int(((i + 1) / sp) * h), int((j / sp) * w):int(((j + 1) / sp) * w)]
                G_blk_list.append(G_blk)
                Gr_blk_list.append(Gr_blk)

        sum = 0
        for i in range(sp * sp):
            mssim = measure.compare_ssim(G_blk_list[i], Gr_blk_list[i])
            sum = mssim + sum

        nrss = 1 - sum / (sp * sp * 1.0)
        return nrss

    # 高斯滤波
    Ir = cv2.GaussianBlur(image, (7, 7), 0)
    G = sobel(image)
    Gr = sobel(Ir)

    blocksize = 8
    ## 获取块信息
    nrss = getBlock(G, Gr)
    return nrss


def brenner(img):
    """
    :param img:ndarray 二维灰度图像
    :return: float 图像约清晰越大
    """
    shape = np.shape(img)
    out = 0
    for x in range(0, shape[0] - 2):
        for y in range(0, shape[1]):
            out += (int(img[x + 2, y]) - int(img[x, y])) ** 2
    return out / (shape[0] * shape[1])


def brightness(color_img):
    if len(color_img.shape) < 3:
        print("color image needed")
        return
    HLS = cv2.cvtColor(color_img, cv2.COLOR_BGR2HLS)
    _, l, _ = cv2.split(HLS)
    return np.mean(l)


if __name__ == "__main__":

    # dataset_path = r'D:\火眼智能\隧道表观\螺栓状态分类\分类测试\20211208\prediction_41\relabeled_002'
    dataset_path = r'C:\Users\zk\Desktop\Project-411_classification\datasets\screw20211206\org\train\001'

    class_list = ["001", "002"]
    dir_list = ["train", "val"]
    # train_path = r'D:\火眼智能\螺栓状态分类\分类测试\樊迎萍重标注'
    # val_path = r'D:\火眼智能\螺栓状态分类\分类测试\樊迎萍重标注'
    # test_path = './datasets/light/defect'

    brightness_list = []
    sharpness_list = []
    height_list = []
    width_list = []

    blur_list = []
    small_list = []
    dark_list = []

    save_path = os.path.join(os.path.dirname(dataset_path), 'clean_data')
    mkdir(save_path)
    mkdir(os.path.join(save_path, 'train', '001'))
    mkdir(os.path.join(save_path, 'train', '002'))
    mkdir(os.path.join(save_path, 'val', '001'))
    mkdir(os.path.join(save_path, 'val', '002'))

    dirty_path = os.path.join(os.path.dirname(dataset_path), 'dirty_data')
    mkdir(os.path.join(dirty_path, 'blur'))
    mkdir(os.path.join(dirty_path, 'small'))
    mkdir(os.path.join(dirty_path, 'dark'))

    # for sub_dir in os.listdir(dataset_path):
    #     if sub_dir == 'meta':
    #         continue
    #     for cls in os.listdir(os.path.join(dataset_path, sub_dir)):
    #         clspath = os.path.join(dataset_path, sub_dir, cls)

    for file in os.listdir(dataset_path):

        image_path = os.path.join(dataset_path, file)
        if not os.path.isfile(image_path):
            continue

        image = Image.open(image_path)
        # image = cv2.imread(os.path.join(test_path, file), -1)
        image_data = np.array(image)

        gray = cv2.cvtColor(image_data, cv2.COLOR_RGB2GRAY) if len(image_data.shape) > 2 else image_data.copy()

        brightness = np.mean(gray)

        # quality_detector = cv2.quality_QualityBRISQUE()
        # quality = quality_detector.compute(img=image_data)

        sharpness = brenner(gray)
        sharpness_list.append(sharpness)

        width_list.append(image.width)
        height_list.append(image.height)
        brightness_list.append(brightness)

        # sharpness
        # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) >= 3 else image
        # gray = image[:, :, 2] if image.shape[2] >= 3 else image
        if image.width < 100 or image.height < 100:
            # cv2.namedWindow('small: '+file, 0)
            # cv2.imshow('small: '+file, image_data)
            # cv2.waitKey()
            # cv2.destroyWindow('small: '+file)

            small_list.append(file)
            # shutil.copy(image_path, os.path.join(dirty_path, 'small'))
            continue

        if brightness < 25:
            # cv2.namedWindow('dark: ' + file, 0)
            # cv2.imshow('dark: ' + file, image_data)
            # cv2.waitKey()
            # cv2.destroyWindow('dark: ' + file)

            dark_list.append(file)
            # shutil.copy(image_path, os.path.join(dirty_path, 'dark'))

            continue

        if sharpness < 5:
            # win_name = 'blur: %f, ' % sharpness + file
            # cv2.namedWindow(win_name, 0)
            # cv2.imshow(win_name, image_data)
            # cv2.waitKey()
            # cv2.destroyWindow(win_name)

            # shutil.copy(image_path, os.path.join(dirty_path, 'blur'))
            blur_list.append(file)
            continue

        # shutil.copy(image_path, os.path.join(save_path, sub_dir, cls, file))

    print('图像高度平均值：%f, 最大值: %f, 最小值: %f' % (np.mean(height_list), np.max(height_list), np.min(height_list)))
    print('图像宽度平均值：%f, 最大值: %f, 最小值: %f' % (np.mean(width_list), np.max(width_list), np.min(width_list)))

    print('图像亮度平均值：%f, 最大值: %f, 最小值: %f' % (np.mean(brightness_list), np.max(brightness_list), np.min(brightness_list)))
    print('图像清晰度平均值：%f, 最大值: %f, 最小值: %f' % (np.mean(sharpness_list), np.max(sharpness_list), np.min(sharpness_list)))

    print('图像尺寸过小：%d' % len(small_list))
    print('图像过暗：%d' % len(dark_list))
    print('图像模糊：%d' % len(blur_list))
