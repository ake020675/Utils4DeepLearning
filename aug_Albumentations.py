"""
测试目标检测增广工具Albumentations
    对注浆孔002大图进行增广（带bbox: [xmin, ymin, xmax, ymax]）
"""

import os
import random
import xml.etree.ElementTree as ET

import albumentations as A
import cv2
import numpy as np
from PIL import Image

BOX_COLOR = (255, 0, 0)  # Red
TEXT_COLOR = (255, 255, 255)  # White


# bbox： four formats:
# pascal_voc: [x_min, y_min, x_max, y_max]
# albumentations: normalized [x_min, y_min, x_max, y_max]
# coco: [x_min, y_min, width, height]
# yolo: [x_center, y_center, width, height]

def visualize_bbox(img, bbox, class_name, color=BOX_COLOR, thickness=2):
    """Visualizes a single bounding box on the image"""
    # x_min, y_min, w, h = bbox
    # x_min, x_max, y_min, y_max = int(x_min), int(x_min + w), int(y_min), int(y_min + h)
    # x_min, y_min, x_max, y_max = bbox
    x_min, y_min, x_max, y_max = np.array(bbox, dtype='int')[:]  # float2int

    # cv2.rectangle(np.array(img), (x_min, y_min), (x_max, y_max), color=BOX_COLOR, thickness=thickness)
    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color=BOX_COLOR, thickness=thickness)

    ((text_width, text_height), _) = cv2.getTextSize(class_name, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)
    cv2.rectangle(img, (x_min, y_min - int(1.3 * text_height)), (x_min + text_width, y_min), BOX_COLOR, -1)
    cv2.putText(
        img,
        text=class_name,
        org=(x_min, y_min - int(0.3 * text_height)),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=3.5,
        color=TEXT_COLOR,
        lineType=cv2.LINE_AA,
    )
    return img


def visualize(image, bboxes, category_names):
    img = image.copy() if len(image.shape) > 2 else cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    for bbox, category_name in zip(bboxes, category_names):
        # class_name = category_id_to_name[category_id]
        img = visualize_bbox(img, bbox, category_name)
    # plt.figure(figsize=(12, 12))
    # plt.axis('off')
    # plt.imshow(img)
    # plt.savefig('vis_transformed.png')
    return img


# def visualize(image, bboxes, category_ids, category_id_to_name):
#     img = image.copy() if len(image.shape) > 2 else cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
#
#     for bbox, category_id in zip(bboxes, category_ids):
#         class_name = category_id_to_name[category_id]
#         img = visualize_bbox(img, bbox, class_name)
#     plt.figure(figsize=(12, 12))
#     plt.axis('off')
#     plt.imshow(img)


def get_bboxes_from_xml(xml_path):
    tree = ET.ElementTree(file=xml_path)
    root = tree.getroot()

    bboxes = []
    object_names = []
    for elem in root.iter(tag='object'):
        # 获取矩形框坐标
        box = [int(el.text) for el in elem.find('bndbox')]
        bboxes.append(box)
        object_name = elem.find('name').text
        object_names.append(object_name)

    return bboxes, object_names


def save_new_xml(xml_path, new_xml_path, new_boxes):
    tree = ET.ElementTree(file=xml_path)
    root = tree.getroot()

    xml_filename = root.find('filename')
    if '/' or r"\\" in new_xml_path:
        xml_filename.text = os.path.basename(new_xml_path).split('.')[0]
    else:
        xml_filename.text = new_xml_path.split('.')[0]

    # bboxes = []
    # object_names = []
    ObjectSet = root.findall('object')  # 找到文件中所有含有object关键字的地方
    for i, Object in enumerate(ObjectSet):
        x = Object.find('bndbox')
        new_box = np.array(new_boxes[i], dtype='int')[:]
        x.find('xmin').text = str(new_box[0])  # -1
        x.find('ymin').text = str(new_box[1])  # -1
        x.find('xmax').text = str(new_box[2])  # -1
        x.find('ymax').text = str(new_box[3])  # -1

    tree.write(new_xml_path + '.xml')


def tranform_with_box(image, boxes=None, category_ids=None, augment_type='hflip'):
    # Define an augmentation pipeline
    if augment_type == 'hflip':
        transform = A.Compose(
            [A.HorizontalFlip(p=1)],
            bbox_params=A.BboxParams(format='pascal_voc', label_fields=['category_ids']),  # voc格式
        )
    if augment_type == 'vflip':
        transform = A.Compose(
            [A.VerticalFlip(p=1)],
            bbox_params=A.BboxParams(format='pascal_voc', label_fields=['category_ids']),  # voc格式
        )

    if augment_type == 'Flip':
        transform = A.Compose(
            [A.Flip(p=1)],
            bbox_params=A.BboxParams(format='pascal_voc', label_fields=['category_ids']),  # voc格式
        )

    if augment_type == 'rotate180':
        transform = A.Compose(
            [A.HorizontalFlip(p=1),
             A.VerticalFlip(p=1)],
            bbox_params=A.BboxParams(format='pascal_voc', label_fields=['category_ids']),  # voc格式
            # p=0.5
        )

    random.seed(7)
    transformed = transform(image=image, bboxes=boxes, category_ids=category_ids)
    return transformed


# 仅增广原图
def tranform_without_box(image, augment_type='hflip'):
    # Define an augmentation pipeline
    if augment_type == 'hflip':
        transform = A.HorizontalFlip(p=1)

    if augment_type == 'vflip':
        transform = A.VerticalFlip(p=1)

    if augment_type == 'Flip':
        transform = A.Flip(p=1)

    if augment_type == 'rotate180':
        transform = A.Compose(
            [A.HorizontalFlip(p=1),
             A.VerticalFlip(p=1)],
        )

    if augment_type == 'GaussianBlur':
        transform = A.GaussianBlur(blur_limit=(15, 15), p=1)  # voc格式

    if augment_type == 'pad_rotate90':
        transform = A.Compose(
            [A.PadIfNeeded(min_height=256, min_width=256, p=1),
             A.Rotate(limit=90, p=1)],
        )

    if augment_type == 'CLAHE':
        transform = A.CLAHE(clip_limit=4.0, p=1)  # voc格式

    if augment_type == 'ColorJitter':
        transform = A.ColorJitter(p=1)  # voc格式

    if augment_type == 'RandomBrightnessContrast':
        transform = A.RandomBrightnessContrast(brightness_limit=(0.1, 0.3),
                                               contrast_limit=(0.1, 0.3), p=1)

    if augment_type == 'blur':
        transform = A.OneOf(
            [A.Blur(blur_limit=7, p=1),
             A.MedianBlur(blur_limit=7, p=1)
             ],  # voc格式
        )

    if augment_type == 'JpegCompression':
        transform = A.JpegCompression(quality_lower=85, quality_upper=95, p=1)

    if augment_type == 'Superpixels':
        transform = A.Superpixels(p_replace=0.1, n_segments=50, max_size=100, p=1)

    if augment_type == 'ISONoise':
        transform = A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=1)  # rgb,int8

    if augment_type == 'GaussNoise':
        transform = A.GaussNoise(var_limit=(10, 50), mean=0, p=1)

    if augment_type == 'Sharpen':
        transform = A.Sharpen(p=1)

    if augment_type == 'Pad':
        transform = A.PadIfNeeded(min_height=256, min_width=256, border_mode=0, value=0, p=1)  # cv2.BORDER_CONSTANT

    if augment_type == 'Equalize':
        transform = A.Equalize(p=1)  # mode='cv',

    if augment_type == 'Emboss':
        transform = A.Emboss(p=1)

    if augment_type == 'FDA':
        transform = A.FDA(p=1)  # reference image

    if augment_type == 'FancyPCA':
        transform = A.FancyPCA(p=1)  # rgb,int8

    if augment_type == 'MotionBlur':
        transform = A.MotionBlur(blur_limit=7, p=1)

    if augment_type == 'MultiplicativeNoise':
        transform = A.MultiplicativeNoise(p=1)

    if augment_type == 'RandomGamma':
        transform = A.RandomGamma(p=1)

    if augment_type == 'Rotate_theta':
        theta = 15
        transform = A.Rotate(limit=theta,
                             interpolation=cv2.INTER_LINEAR,
                             border_mode=cv2.BORDER_CONSTANT,
                             value=(0, 0, 0),
                             p=0.5, )

    if augment_type == 'RandomScale':
        transform = A.RandomScale(scale_limit=0.2, interpolation=1, p=0.5)

    random.seed(7)
    transformed = transform(image=image)
    return transformed


def voc2coco(bbox):
    return [bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]]  # x,y,h,w


def coco2voc(bbox):
    return [bbox[0], bbox[1], bbox[2] + bbox[0], bbox[3] + bbox[1]]  # x1,y1,x2,y2


if __name__ == '__main__':
    input_dir = r'C:\Users\zk\Desktop\Project-411_classification\datasets\screw20211206\relabeled\train\002'

    # # 批量增广图像
    # save_dir = input_dir + '_augment'
    # if not os.path.exists(save_dir):
    #     os.makedirs(save_dir)
    # # augment_list = ["vflip", "hflip", "rotate180"]
    # augment_list = ["Pad"]
    # augment_task = "classification"  # {"classification", "detection"}
    # for file in os.listdir(input_dir):
    #     if augment_task == "detection":
    #         if file.endswith('.xml'):  # 目标检测：存在标注xml
    #             filename = file.split('.')[0]
    #             image = np.array(Image.open(os.path.join(input_dir, filename + '.png')))
    #             xml_path = os.path.join(input_dir, filename + '.xml')
    #             boxes, object_names = get_bboxes_from_xml(xml_path)
    #             category_ids = [int(name) for name in object_names]
    #
    #             for augment_type in augment_list:
    #                 save_name = filename + '_' + augment_type
    #                 transformed = tranform_with_box(image=image, boxes=boxes, category_ids=category_ids,
    #                                                 augment_type=augment_type)
    #                 save_new_xml(xml_path=xml_path, new_boxes=transformed["bboxes"],
    #                              new_xml_path=os.path.join(save_dir, save_name))  # 保存增广xml
    #                 Image.fromarray(transformed["image"]).save(os.path.join(save_dir, save_name + '.png'))  # 保存增广图像
    #     else:  # 针对分类的增广
    #         image = np.array(Image.open(os.path.join(input_dir, file)))
    #         filename, ext = os.path.splitext(file)
    #         for augment_type in augment_list:
    #             transformed = tranform_without_box(image=image, augment_type=augment_type)
    #             Image.fromarray(transformed["image"]).save(os.path.join(save_dir, filename+'_'+augment_type + ext))  # 保存增广图像

    # 测试单个文件带box增广
    augment_type = "Equalize"
    filename = '2_3_24_4_0'
    image = np.array(Image.open(os.path.join(input_dir, filename + '.jpg')))
    xml_path = os.path.join(input_dir, filename + '.xml')
    save_name = filename + '_' + augment_type

    if os.path.exists(xml_path):  # 获取标注信息并显示
        # 带bbox增广
        boxes, object_names = get_bboxes_from_xml(xml_path)
        category_ids = [int(name) for name in object_names]

        vis_org = visualize(
            image,
            boxes,
            category_names=object_names,
        )
        cv2.namedWindow('org_im', 0)
        cv2.imshow('org_im', vis_org)
        cv2.waitKey()

        if boxes is not None:
            transformed = tranform_with_box(image=image, boxes=boxes, category_ids=category_ids,
                                            augment_type=augment_type)
            vis_transformed = visualize(
                transformed[0],
                transformed[1],
                category_names=object_names,
            )
            cv2.namedWindow('vis_transformed', 0)
            cv2.imshow('vis_transformed', vis_transformed)
            cv2.waitKey()

    else:  # 只增广图像
        transformed = tranform_without_box(image=image, augment_type=augment_type)
        cv2.namedWindow('image', 0)
        cv2.imshow('image', image)
        cv2.namedWindow('transformed', 0)
        cv2.imshow('transformed', transformed['image'])
        cv2.waitKey()

    # 保存新的box标注信息
    # save_new_xml(xml_path, save_name, transformed[1])
    # Image.fromarray(transformed[0]).save(os.path.join(save_dir, save_name + '.png'))  # 保存增广图像
