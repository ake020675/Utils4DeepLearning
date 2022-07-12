"""
  对原始数据库进行分割等预处理，构建数据库
  in : images
  out: dataset
"""

import os
from shutil import copy
import random
from aug_Augmentor import img_aug


def mkfile(filepath):
    if not os.path.exists(filepath):
        os.makedirs(filepath)

# GroutingWell
# dataset_path = 'datasets/GroutingWell/enhanced_images'
# input_path = r'datasets/GroutingWell/enhanced_images'

# screw
# dataset_path = 'datasets/screw20211116/'
# input_path = r'D:\火眼智能\螺栓状态分类\分类测试\20211112\樊迎萍重标注\018_019'

# # class_labels = [cla for cla in os.listdir(input_path) if ".txt" not in cla]
# class_labels = ["001", "002"]
#
# mkfile(dataset_path + '/train')
# for cla in class_labels:
#     mkfile(dataset_path + '/train/' + cla)
#
# mkfile(dataset_path + '/val')
# for cla in class_labels:
#     mkfile(dataset_path + '/val/' + cla)
#
# split_rate = 0.2
#
# for cla in class_labels:  # cla: real/fake
#     cla_path = os.path.join(input_path, cla)
#     filenames = os.listdir(cla_path)
#     cla_filelist = []
#
#     # 获取每类下所有图像文件路径
#     for file in filenames:  # real/fake下每个文件/文件夹
#         imgdir = os.path.join(cla_path, file)
#         if os.path.isdir(imgdir):  # 若cla下的文件为目录：将目录内所有图像文件加入list
#             images = os.listdir(imgdir)
#             if images is None:
#                 continue
#             else:
#                 for image in images:
#                     img_path = os.path.join(imgdir, image)
#                     cla_filelist.append(img_path)
#         else:  # 若cla下的文件为文件：将目录内所有图像文件加入list
#             cla_filelist.append(imgdir)
#
#     num = len(cla_filelist)
#     eval_index = random.sample(cla_filelist, k=int(num * split_rate))  # 随机采样
#     for index, image in enumerate(cla_filelist):
#         if image in eval_index:  # 验证集
#             name, extname = os.path.splitext(os.path.basename(image))
#             newname = name + '.png'
#             new_path = dataset_path + '/val/' + cla + "/" + newname
#             copy(image, new_path)
#         else:  # 训练集
#             name, extname = os.path.splitext(os.path.basename(image))
#             newname = name + '.png'
#             new_path = dataset_path + '/train/' + cla + "/" + newname
#             copy(image, new_path)
#         print("/r[{}] processing [{}/{}] /n".format(cla, index + 1, num), end="")  # processing bar
#     print()
#
# print("processing done!")


# # 根据包含018\019类标注的大图列表，查找对应的所有目标检测结果，并对其重新标注，得到真正的018_019类小图
# # 输入：包含018\019类目标框标注信息的大图列表
# # 输出：大图对应的所有小图（目标检测结果）
# screws_detection_dir = r'./datasets/screw20210901/test_20211012'  # 目标检测得到的小图
# imlist_018_019 = r'D:\火眼智能\螺栓状态分类\目标检测结果\images_018_019.txt'  # 包含018、019类标注的大图
#
# save_dir = os.path.join(screws_detection_dir, 'detections_include_018_019')
# if not os.path.exists(save_dir):
#     mkfile(save_dir)
#
# f = open(imlist_018_019, 'r')
# for line in f.readlines():
#     imname = line.split('.png')[0]
#     for cls in os.listdir(screws_detection_dir):  # 001, 002
#
#         for filename in os.listdir(os.path.join(screws_detection_dir, cls)):
#             if imname in filename:
#                 if not os.path.exists(os.path.join(save_dir, filename)):
#                     copy(os.path.join(screws_detection_dir, cls, filename), save_dir)


# voc数据格式划分
# from glob import glob
# from sklearn.model_selection import train_test_split
#
# # 1.结果保存路径
# saved_path = "experiment/test/results_screw/"  # 保存路径
#
# # 2.创建要求文件夹
# if not os.path.exists(saved_path + "ImageSets/Main/"):
#     os.makedirs(saved_path + "ImageSets/Main/")
#
# # 3.split files for txt
# txtsavepath = saved_path + "ImageSets/Main/"
# ftrainval = open(txtsavepath + '/trainval.txt', 'w')
# ftrain = open(txtsavepath + '/train.txt', 'w')
# fval = open(txtsavepath + '/val.txt', 'w')
#
# # total_files = glob(saved_path + "Annotations/*.xml")
#
# # total_files = [i.split("/")[-1].split(".xml")[0] for i in total_files]
#
# for file in total_files:
#     ftrainval.write(file + "\n")
#
# # split
# train_files, val_files = train_test_split(total_files, test_size=0.15, random_state=42)
#
# # train
# for file in train_files:
#     ftrain.write(file + "\n")
# # val
# for file in val_files:
#     fval.write(file + "\n")
#
# ftrainval.close()
# ftrain.close()
# fval.close()
