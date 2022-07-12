"""
  对新增图片进行分割，加入screw训练和测试集
  in : 新增图片路径
  out: 数据集
"""

import os
from shutil import copy
import random
from aug_Augmentor import img_aug


def mkfile(filepath):
    if not os.path.exists(filepath):
        os.makedirs(filepath)


# screw
dataset_path = 'datasets/screw20211116/'
input_path = r'D:\火眼智能\螺栓状态分类\分类测试\20211112\樊迎萍重标注'  # new data

# class_labels = [cla for cla in os.listdir(input_path) if ".txt" not in cla]
class_labels = ["018_019"]

mkfile(dataset_path + '/train')
for cla in class_labels:
    mkfile(dataset_path + '/train/' + cla)

mkfile(dataset_path + '/test')
for cla in class_labels:
    mkfile(dataset_path + '/test/' + cla)

split_rate = 0.2

for cla in class_labels:  # cla: real/fake
    cla_path = os.path.join(input_path, cla)
    filenames = os.listdir(cla_path)

    num = len(filenames)
    eval_index = random.sample(filenames, k=int(num * split_rate))  # 随机采样
    for index, image in enumerate(filenames):
        if image in eval_index:  # 验证集
            name, extname = os.path.splitext(os.path.basename(image))
            newname = name + '.png'
            new_path = dataset_path + '/val/' + cla + "/" + newname
            copy(image, new_path)
        else:  # 训练集
            name, extname = os.path.splitext(os.path.basename(image))
            newname = name + '.png'
            new_path = dataset_path + '/train/' + cla + "/" + newname
            copy(image, new_path)
        print("/r[{}] processing [{}/{}] /n".format(cla, index + 1, num), end="")  # processing bar
    print()

print("processing done!")

