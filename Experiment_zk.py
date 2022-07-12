# coding:utf-8
"""
by zk
用于隧道单类部件检测
从图像文件和xml标注文件，创建VOC格式目标检测数据集
    1.xml标注文件转换为txt标注文件
    2.创建train/val/test.txt文件
"""

import os
import random
import xml.etree.ElementTree as ElementTree
import sys
import glob
import shutil
import cv2
from pathlib import Path


FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative


def convert_bbox_coco2voc(size, box):
    dw = 1. / (size[0])
    dh = 1. / (size[1])
    x = (box[0] + box[1]) / 2.0 - 1
    y = (box[2] + box[3]) / 2.0 - 1
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return x, y, w, h


def convert_annotations_xml2txt(xml_path, classes, cls_dataset, im_dir=None):
    """
    将xml标注（coco格式）转换为txt（voc格式）,抛弃非cls_dirname类的object标注
    :param xml_path: xml标注文件路径
    :param classes: 所有标注类别
    :param cls_dataset: 待转换的目标类别
    :param im_dir: 原始图像文件夹，用于save crop
    :return:
    """

    if im_dir is None:
        im_dir = os.path.dirname(xml_path) + '/img'
    if not os.path.exists(im_dir):
        print('im_dir not exists! cannot save crop!')

    txt_path = os.path.dirname(xml_path) + '/labels'
    if not os.path.exists(txt_path):
        os.makedirs(txt_path)

    crop_path = os.path.dirname(xml_path) + '/crops'
    if not os.path.exists(crop_path):
        os.makedirs(crop_path)

    object_numbers = 0
    for xml_file in os.listdir(xml_path):

        file_name = xml_file.split('.')[0]
        save_file = os.path.join(txt_path,  file_name + '.txt')
        in_file = os.path.join(xml_path, xml_file)
        out_file = open(save_file, 'w', encoding='UTF-8')
        if os.path.getsize(in_file) == 0:
            print(f'xml error: {in_file}')
            continue

        tree = ElementTree.parse(in_file)
        root = tree.getroot()
        size = root.find('size')
        w = int(size.find('width').text)
        h = int(size.find('height').text)
        objs = root.iter('object')
        for i, obj in enumerate(objs):

            # drop difficult
            difficult = obj.find('difficult').text
            if int(difficult) == 1:
                continue

            # drop objects of other class
            obj_cls = obj.find('name').text
            if obj_cls != cls_dataset:  # or obj_cls not in classes
                continue

            object_numbers += 1

            # get bndbox
            cls_id = classes.index(obj_cls)
            xmlbox = obj.find('bndbox')
            b1, b2, b3, b4 = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text),
                              float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))

            # save crop image of object box
            src_im = cv2.imread(os.path.join(im_dir, file_name + '.png'), 1)
            save_crop = os.path.join(crop_path, file_name + '_' + str(object_numbers) + '.png')
            cv2.imwrite(save_crop, src_im[int(b3):int(b4), int(b1):int(b2)])

            # x1x2y1y2 to xywh
            # 标注越界修正
            if b2 > w:
                b2 = w
            if b4 > h:
                b4 = h
            b = (b1, b2, b3, b4)
            bb = convert_bbox_coco2voc((w, h), b)
            out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')
        out_file.close()


def create_labels(dataset_path='W:/lm/yolov5-master/guizhen', xml_path=None):
    """
    将xml格式标注（yolo）转换为txt(voc)，存入labels文件夹：
    """
    classes = [str(os.path.basename(dataset_path))]  # "001"
    if xml_path is None:
        xml_path = os.path.join(dataset_path, 'xml')
    cls_name = os.path.basename(dataset_path)
    convert_annotations_xml2txt(xml_path, classes, cls_name)


def gen_trainval_txt(dataset_path=r'W:/Users/huoyanhou/Desktop/datasets/single/025'):
    """
    生成VOC文件列表train/val/test.txt
    """
    trainval_percent = 1.0
    train_percent = 0.9
    label_path = os.path.join(dataset_path, 'labels')  # 'input xml path'
    txt_path = os.path.join(dataset_path, 'ImageSets/Main')  # 'output txt path'
    total_xml = os.listdir(label_path)
    if not os.path.exists(txt_path):
        os.makedirs(txt_path)

    num = len(total_xml)
    list_index = range(num)
    tv = int(num * trainval_percent)
    tr = int(tv * train_percent)
    trainval = random.sample(list_index, tv)
    train = random.sample(trainval, tr)

    file_trainval = open(txt_path + '/trainval.txt', 'w')
    file_test = open(txt_path + '/test.txt', 'w')
    file_train = open(txt_path + '/train.txt', 'w')
    file_val = open(txt_path + '/val.txt', 'w')

    for i in list_index:
        name = total_xml[i][:-4]
        image_path = os.path.join(dataset_path, 'images', name + '.png')
        if i in trainval:
            file_trainval.write(image_path + '\n')
            if i in train:
                file_train.write(image_path + '\n')
            else:
                file_val.write(image_path + '\n')
        else:
            file_test.write(image_path + '\n')

    file_trainval.close()
    file_train.close()
    file_val.close()
    file_test.close()


def gen_test_txt(dataset_path=r'C:\Users\zk\Desktop\yolov5-v6.1\yolov5-master\test_20220510_024-039\024',
                 image_path=None):
    """
    为单类部件测试图像生成文件列表test.txt
    """
    if image_path is None:
        image_path = os.path.join(dataset_path, 'images')
    label_path = os.path.join(dataset_path, 'labels')

    # 测试列表：image_path内文件
    test_txt = open(os.path.join(dataset_path, 'test.txt'), 'w')  # test txt for cls
    for file in os.listdir(image_path):
        imname = file.split('.')[0]
        test_txt.write(os.path.join(image_path, imname + '.png') + '\n')  # image list of label_path
    test_txt.close()

    # 验证列表：label_path内文件
    trainval_txt = open(os.path.join(dataset_path, 'trainval.txt'), 'w')  # trainval txt for cls
    for file in os.listdir(label_path):
        imname = file.split('.')[0]
        trainval_txt.write(os.path.join(image_path, imname + '.png') + '\n')  # image list of image_path
    trainval_txt.close()

    print('all txt generated!')


def check_imdir(dataset_path):
    """
    检查数据集内图像文件夹是否为'images'，若否则重命名
    """
    im_dir = os.path.join(dataset_path, 'images')
    if not os.path.exists(im_dir):
        os.rename(os.path.join(dataset_path, 'img'), im_dir)


def select_test_images(dataset_path, test_dir='/test_imgs'):
    label_path = os.path.join(dataset_path, 'labels')

    # 测试列表：image_path内文件
    test_txt = open(os.path.join(dataset_path, 'test.txt'), 'w')  # test txt for cls
    for file in random.sample(os.listdir(label_path), 2):
        imname = file.split('.')[0]


def main():
    """
    处理隧道单部件检测：准备数据集/训练模型/转换模型格式/推理测试
    :return:
    """
    try:
        # dir_multi_class = r'W:/Users/huoyanhou/Desktop/datasets/single2'
        dir_multi_class = r'C:\Users\zk\Desktop\yolov5-v6.1\yolov5-master\test_20220510_024-039'
        src_xmlpath = os.path.join(dir_multi_class, 'xml')

        # classes = os.listdir(dir_multi_class)
        # classes = ["%03d" % i for i in (range(24, 40))]
        classes = ["%03d" % i for i in (range(1, 3))]
        # classes = ['026']
        for cls in classes:

            # 为每类部件创建数据集文件夹
            dataset_path = os.path.join(dir_multi_class, 'single/'+cls)
            if not os.path.exists(dataset_path):
                os.makedirs(dataset_path)

            # 准备测试数据集
            xmlpath = os.path.join(dataset_path, 'xml')
            if not os.path.exists(xmlpath):
                os.makedirs(xmlpath)
                # copy xml
                xmls_cls = glob.glob(os.path.join(src_xmlpath, cls+'/*.xml'))
                for xml_cls in xmls_cls:
                    shutil.copy(xml_cls, xmlpath)

            # 生成labels: xml2txt, save cropped image
            label_path = os.path.join(dataset_path, 'labels')
            if not os.path.exists(label_path):
                os.makedirs(label_path)
                convert_annotations_xml2txt(xmlpath, classes=[cls], cls_dataset=cls,
                                            im_dir=os.path.join(dir_multi_class, 'images'))
            # 生成test.txt
            if not os.path.exists(os.path.join(dataset_path, 'test.txt')):
                gen_test_txt(dataset_path, image_path=os.path.join(dir_multi_class, 'images'))
            # 每类选择2幅图像用于测试
            test_dir = dir_multi_class + '/test_imgs'
            if not os.path.exists(test_dir):
                os.makedirs(test_dir)
            for file in random.sample(os.listdir(label_path), 2):
                imname = file.split('.')[0]
                impath = os.path.join(dir_multi_class, 'images/' + imname + '.png')
                shutil.copy(impath, test_dir)

            # gen_data_yaml
            data_yaml = os.path.join(dataset_path, 'data.yaml')
            if not os.path.exists(data_yaml):
                with open(data_yaml, 'w') as f:
                    # f.write('train: {}\n'.format(os.path.join(dataset_path, 'train.txt')))
                    f.write('val: {}\n'.format(os.path.join(dataset_path, 'trainval.txt')))
                    f.write('test: {}\n'.format(os.path.join(dataset_path, 'test.txt')))
                    f.write('nc: 1\n')
                    f.write('names: ["{}"]'.format(cls))

            # # 准备训练数据集
            # check_imdir(dataset_path)  # 检查images文件夹
            # create_labels(dataset_path, xml_path)  # 生成txt标注文件，存放于Labels文件夹
            # gen_trainval_txt(dataset_path)  # 生成train/val/test.txt

            # # 部件类别、模型参数
            # class_names = [str(cls)]
            # best_weight = os.path.join('runs/train', str(cls) + '/weights/best.pt')  #
            # input_sz = [640, 640]
            # torchscript_model = os.path.splitext(best_weight)[0] + '.torchscript'

            # # 训练
            # if not os.path.exists(best_weight):
            #     import train
            #     train.run(data=dataset_path+'/data.yaml',
            #               name=str(cls),
            #               # imgsz=input_sz
            #               )
            #
            # # 转换模型
            # if not os.path.exists(torchscript_model):
            #     import export
            #     export.run(data=dataset_path+'/data.yaml', weights=best_weight,
            #                # imgsz=input_sz,
            #                include=('torchscript'))

            # # 验证模型、统计数据
            # import val
            # val.run(data=dataset_path+'/data.yaml',
            #         weights=best_weight,
            #         name='026',
            #         device=0)

            # # 推理
            # import detect
            # detect.run(source='path/', weights=best_weight,
            #            classes=class_names, name=str(cls),
            #            imgsz=input_sz)

    except Exception as e:
        print(f'{e}')


if __name__ == "__main__":
    main()
