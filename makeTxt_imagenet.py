"""
生成imagenet分类数据集标签 txt
by zk
创建：20211117
最后修改：20220302
"""

import os
import glob
import random
from sklearn.model_selection import train_test_split


# 按文件夹名分配类别标签
def generate_txt(images_dir, class_map_dict):
    # 读取所有文件名
    imgs_dirs = glob.glob(images_dir + "/*/*")
    meta_dir = os.path.join(os.path.dirname(images_dir), "meta")
    if not os.path.exists(meta_dir):
        os.makedirs(meta_dir)

    # 打开写入文件
    typename = os.path.basename(images_dir)
    target_txt_path = os.path.join(meta_dir, typename + ".txt")  # train/val/test
    f = open(target_txt_path, "w")
    # 遍历所有图片名
    for img_dir in imgs_dirs:
        # 获取第一级目录名称
        # filename = img_dir.split("/")[-2]
        relative_path = img_dir.split(typename)[1]
        filename = os.path.basename(relative_path)

        cls_name = os.path.basename(os.path.dirname(relative_path))
        cls_label = class_map_dict[cls_name]  # 根据文件夹名分配类别标签
        # 写入文件
        # relate_path = re.findall(typename + "/([\w / - .]*)", img_dir)
        f.write(cls_name + '/' + filename + " " + cls_label + "\n")


def generate_txt_trainval_test(images_dir, class_map_dict):
    # 读取所有文件名
    imgs_dirs = glob.glob(images_dir + "/*/*")
    meta_dir = os.path.join(os.path.dirname(images_dir), "meta")
    if not os.path.exists(meta_dir):
        os.makedirs(meta_dir)

    # 打开写入文件
    typename = os.path.basename(images_dir)
    target_txt_path = os.path.join(meta_dir, typename + ".txt")
    f = open(target_txt_path, "w")
    # 遍历所有图片名
    for img_dir in imgs_dirs:
        # 获取第一级目录名称
        # filename = img_dir.split("/")[-2]
        # if typename ==
        relative_path = img_dir.split(typename)[1]
        filename = os.path.basename(relative_path)

        cls_name = os.path.basename(os.path.dirname(relative_path))
        cls_label = class_map_dict[cls_name]  # 根据文件夹名分配类别标签
        # 写入文件
        # relate_path = re.findall(typename + "/([\w / - .]*)", img_dir)
        f.write(cls_name + '/' + filename + " " + cls_label + "\n")


def get_map_dict(classmap_txt):
    # 读取所有类别映射关系
    class_map_dict = {}
    with open(classmap_txt, "r") as F:
        lines = F.readlines()
        for line in lines:
            line = line.split("\n")[0]
            filename, cls, num = line.split(" ")
            class_map_dict[filename] = num
    return class_map_dict


def make_txt_select_n(root_dir, class_map_dict, number_train=500):
    """
    生成训练/验证/测试集对应的标注文件txt（imagenet格式）
    从分类数据集中随机选取固定数量图片作为训练集，其余为验证集和测试集
    Args:
        root_dir: 数据集根目录
        class_map_dict: 类别列表，从txt读取
        number_train: 选取每个类别的训练图片数量
    Returns:

    """
    class_dirs = os.listdir(root_dir)

    train_txt = os.path.join(root_dir, 'meta', "train.txt")
    val_txt = os.path.join(root_dir, 'meta', "val.txt")
    test_txt = os.path.join(root_dir, 'meta', "test.txt")
    f_train = open(train_txt, "w")
    f_val = open(val_txt, "w")
    f_test = open(test_txt, "w")

    for cls in class_dirs:
        if cls == 'meta':
            continue
        im_files = os.listdir(os.path.join(root_dir, cls))
        cls_label = class_map_dict[cls]

        train_files = random.sample(im_files, k=number_train)  # 随机选取500
        val_test_files = [i for i in im_files if i not in train_files]
        val_files, test_files = train_test_split(val_test_files, test_size=0.5, random_state=42)
        for file in train_files:
            f_train.write(cls + '/' + file + " " + cls_label + "\n")
        for file in val_files:
            f_val.write(cls + '/' + file + " " + cls_label + "\n")
        for file in test_files:
            f_test.write(cls + '/' + file + " " + cls_label + "\n")

    f_train.close()
    f_val.close()
    f_test.close()


if __name__ == '__main__':

    # 需要改为您自己的路径
    root_dir = r'C:\Users\zk\Desktop\mmclassification-0.18.0\datasets\LaserLabel'

    # 在该路径下有train,val，meta三个文件夹
    train_dir = os.path.join(root_dir, "train")
    val_dir = os.path.join(root_dir, "val")

    # 在该路径下有trainval, test, meta三个文件夹
    trainval_dir = os.path.join(root_dir, "trainval")
    test_dir = os.path.join(root_dir, "test")

    meta_dir = os.path.join(root_dir, 'meta')
    if not os.path.exists(meta_dir):
        os.makedirs(meta_dir)
    classmap_txt = os.path.join(meta_dir, "classmap.txt")
    assert (os.path.exists(classmap_txt))

    class_map_dict = get_map_dict(classmap_txt)

    # if os.path.exists(trainval_dir):
    #     generate_txt_trainval_test(images_dir=train_dir, class_map_dict=class_map_dict)
    #     generate_txt_trainval_test(images_dir=test_dir, class_map_dict=class_map_dict)
    #
    # else:
    #     generate_txt(images_dir=train_dir, class_map_dict=class_map_dict)
    #     generate_txt(images_dir=val_dir, class_map_dict=class_map_dict)

    # generate_txt(images_dir=train_dir,
    #              class_map_dict=class_map_dict)

    make_txt_select_n(root_dir=root_dir,
                      class_map_dict=class_map_dict)
