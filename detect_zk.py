# YOLOv5 🚀 by Ultralytics, GPL-3.0 license

"""单类部件检测程序"""

import os
import cv2
import numpy as np
import torch
import logging
import glob
import shutil

from models.common import DetectMultiBackend
from utils.general import (non_max_suppression, scale_coords, colorstr, xywh2xyxy)
from utils.plots import Annotator, colors, save_one_box
from utils.metrics import ConfusionMatrix, ap_per_class, box_iou
from utils.datasets import img2label_paths, create_dataloader
from pathlib import Path
from utils.augmentations import Albumentations, augment_hsv, copy_paste, letterbox, mixup, random_perspective


def process_batch(detections, labels, iou_thres, device='cuda:0'):
    """
    Modified from val.process_batch
    Return correct predictions matrix. Both sets of boxes are in (x1, y1, x2, y2) format.
    Arguments:
        detections (Array[N, 6]), x1, y1, x2, y2, conf, class
        labels (Array[M, 5]), class, x1, y1, x2, y2
        iou_thres: iou threshold
        device:
    Returns:
        correct (Array[N, 10]), for 10 IoU levels
    """
    correct = torch.zeros(detections.shape[0], 1, dtype=torch.bool, device=device)
    iou = box_iou(labels[:, 1:], detections[:, :4])
    x = torch.where((iou >= iou_thres) & (labels[:, 0:1] == detections[:, 5]))  # IoU above threshold and classes match
    if x[0].shape[0]:
        matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()  # [label, detection, iou]
        if x[0].shape[0] > 1:
            matches = matches[matches[:, 2].argsort()[::-1]]
            matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
            # matches = matches[matches[:, 2].argsort()[::-1]]
            matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
        matches = torch.from_numpy(matches).to(device)
        correct[matches[:, 1].long()] = matches[:, 2:3] >= iou_thres
    return correct


@torch.no_grad()
def inference_detector(model, im, im0s, conf_thres=0.5, iou_thres=0.3):
    im = torch.from_numpy(im).to('cuda:0')
    im = im.float()  # uint8 to fp16/32
    im /= 255  # 0 - 255 to 0.0 - 1.0
    if len(im.shape) == 3:
        im = im[None]  # expand for batch dim
    pred = model(im)

    # NMS
    pred = non_max_suppression(pred, conf_thres, iou_thres, max_det=10)

    # Process predictions
    for i, det in enumerate(pred):  # per image
        det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0s.shape).round()
    return det


def check_labels(img_paths, label_paths, weights, task='val'):
    """
    todo:使用已训练模型，对数据集进行推理，根据推理结果检查标注中的漏标/错标
    """
    pad = 0.5
    pt = weights.endswith('.pt')
    rect = False if task == 'benchmark' else pt  # square inference for benchmarks
    # task = task if task in ('train', 'val', 'test') else 'val'  # path to train/val/test images
    dataloader = create_dataloader(img_paths,
                                   imgsz=640,
                                   batch_size=1,
                                   stride=32,
                                   single_cls=False,
                                   pad=pad,
                                   rect=rect,
                                   workers=8,
                                   prefix=colorstr(f'{task}: '))[0]


def main():

    source = r'C:\Users\zk\Desktop\yolov5-v6.1\yolov5-master\test_20220510_024-039\images'  # 测试图像路径
    img_paths = glob.glob(os.path.join(source, '*.png'))
    test_class = 'ZhuJiangKong'
    class_names = [test_class]
    label_path = os.path.join(r'C:\Users\zk\Desktop\yolov5-v6.1\yolov5-master\test_20220510_024-039\single',
                              test_class + '/labels')
    label_paths = glob.glob(
        os.path.join(label_path, '*.txt'))

    # weight, device
    weights = f'runs/train/{test_class}/weights/best.pt'  # 固化好的模型
    # weights = f'runs/train/new{test_class}/weights/best.pt'  # 固化好的模型
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    if not os.path.exists(weights):
        print('weights not exists')
        return

    # 加载模型
    if str(weights).endswith('.torchscript'):
        model = torch.jit.load(weights)
    else:
        model = DetectMultiBackend(weights, device=device)

    # NMS params
    conf_thres = 0.8
    iou_thres = 0.1

    # log
    log_path = f'runs/detect/{test_class}/log.txt'
    # logger = logging.getLogger(log_path)
    mode = 'w'  # if not os.path.exists(log_path) else 'a'
    logging.basicConfig(filename=log_path, filemode=mode, level=logging.INFO)

    # 推理结果保存路径
    class_save_dir = f'runs/detect/{test_class}/' + f'/{test_class}_conf{conf_thres}'  # 含标注的图像测试结果
    if not os.path.exists(class_save_dir):
        os.makedirs(class_save_dir)

    # 虚检验证：对不包含当前类别标注的图像进行检测
    detect_Nonclass = False
    class_image_list = []
    if detect_Nonclass:
        class_label_list = os.listdir(os.path.dirname(source) + f'/single/{test_class}/labels')
        class_image_list = [os.path.join(source, label_file.split('.txt')[0] + '.png') for label_file in
                            class_label_list]
        nonclass_save_dir = f'runs/detect/{test_class}/' + f'/Non{test_class}_conf{conf_thres}'  # 用于保存不含物体的图像测试结果
        if not os.path.exists(nonclass_save_dir):
            os.makedirs(nonclass_save_dir)

    # 标注错误检测
    check_labels = False
    if len(label_paths) == 0:
        print('label_paths not exists! Cannot check labels')
        check_labels = False
    number_total_labels = 0
    number_images_labeled = len(label_paths)
    if check_labels:
        missing_label_list = list()
        error_label_list = list()
        missing_label_path = os.path.join(f'runs/detect/{test_class}', f'误检_{conf_thres}')
        error_label_path = os.path.join(f'runs/detect/{test_class}', f'漏检_{conf_thres}')
        if not os.path.exists(missing_label_path):
            os.makedirs(missing_label_path)
        if not os.path.exists(error_label_path):
            os.makedirs(error_label_path)

    # 测试图像列表
    pred_conf_class = []
    pred_conf_nonclass = []

    # 验证
    jdict, stats, ap, ap_class = [], [], [], []
    seen = 0
    confusion_matrix = ConfusionMatrix(nc=1)  # 单类
    nc = 1
    plots = True
    names = {k: v for k, v in enumerate(model.names if hasattr(model, 'names') else model.module.names)}
    dt, p, r, f1, mp, mr, map50, map = [0.0, 0.0, 0.0], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    for img_path in img_paths:

        # read image
        Im = cv2.imread(img_path)
        height, width = Im.shape[:2]
        org_shape = Im.shape

        # check label exists
        image_name = os.path.basename(img_path)
        label_file = os.path.join(label_path, image_name.split('.')[0] + '.txt')
        check_label = True if check_labels and os.path.exists(label_file) else False
        label_number = 0
        if check_label is True:
            f = open(label_file, 'r', encoding='utf-8')
            labels = f.readlines()
            label_number = len(labels)
            number_total_labels += label_number

        # Preprocess code from detect.py->datasets.LoadImages
        # Padded resize
        input_img = letterbox(Im)[0]
        # Convert
        img = input_img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)

        # to tensor
        Img = torch.from_numpy(img).to(device)
        Img = Img.float()  # uint8 to fp16/32
        Img /= 255  # 0 - 255 to 0.0 - 1.0
        if len(Img.shape) == 3:
            Img = Img[None]  # expand for batch dim

        # inference
        # out = model(Img)

        out = model(Img, augment=True)  # multi scale inference
        # out, train_out = model(Img, val=True)  # inference, loss outputs

        # NMS
        pred = non_max_suppression(out, conf_thres, iou_thres, max_det=100)

        # # adjust coords according to orginal iamge shape
        # for i, det in enumerate(pred):  # per image
        #     det[:, :4] = scale_coords(Img.shape[2:], det[:, :4], Im.shape).round()
        # annotator = Annotator(Im, line_width=line_thickness, example=str(class_names))

        # adjust coords according to orginal iamge shape
        for i, det in enumerate(pred):  # per image
            det[:, :4] = det[:, :4]

        # 保存图像设置
        save_img = True
        hide_labels = False
        hide_conf = False
        save_crop = False
        line_thickness = 3
        if detect_Nonclass and img_path not in class_image_list:
            save_dir = nonclass_save_dir  # 不含标注物体的图像测试结果保存路径
        else:
            save_dir = class_save_dir  # 结果保存路径

        # if det.shape[0] == 0:  # no preds
        #     if img_path in class_image_list and check_label:
        #         # print(f'no targets detected in {image_name}')
        #         error_label_list.append(image_name)
        #         shutil.copy(img_path, error_label_path)
        #     save_img = False
        #     save_crop = False

        if save_img:
            annotator = Annotator(input_img, line_width=line_thickness, example=str(class_names))
            # Write results
            idx_crop = 0
            for *xyxy, conf, index_cls in reversed(det):

                if img_path in class_image_list:
                    pred_conf_class.append(conf.cpu())
                else:
                    pred_conf_nonclass.append(conf.cpu())

                c = int(index_cls)
                label = None if hide_labels else (class_names[c] if hide_conf else f'{class_names[c]} {conf:.2f}')
                annotator.box_label(xyxy, label, color=colors(c, True))
                if save_crop:
                    # file = 'paper_data1/result/crops/01/'
                    crop_file = Path[save_dir] / 'crops' / f'{Path[img_path].stem}_{idx_crop}.jpg'
                    save_one_box(xyxy, input_img,
                                 file=crop_file, BGR=True)
                idx_crop += 1

            im0 = annotator.result()
            cv2.imwrite(os.path.join(save_dir, image_name), im0)

            # check labels with trained model
            if check_label:
                pred_number = det.shape[0]
                if pred_number > label_number:  # 漏标
                    missing_label_list.append(image_name)
                    savePath = os.path.join(missing_label_path, image_name)
                    cv2.imwrite(savePath, im0)
                elif pred_number > label_number:  # 误标
                    error_label_list.append(image_name)
                    savePath = os.path.join(error_label_path, image_name)
                    cv2.imwrite(savePath, im0)

        # # val
        # # todo: load targets
        # targets = targets.to(device)  # 标签中的目标框
        # targets[:, 2:] *= torch.tensor((width, height, width, height), device=device)  # to pixels
        #
        # # Metrics
        # # for si, pred in enumerate(out):
        # labels = targets[1:]
        # nl, npr = labels.shape[0], pred.shape[0]  # number of labels, predictions
        # path, shape = imgPath, Img.shape  # 图像路径， 形状
        # correct = torch.zeros(npr, 1, dtype=torch.bool, device=device)  # init
        # seen += 1
        #
        # if npr == 0:  # 无预测
        #     if nl:  # 有标签
        #         stats.append((correct, *torch.zeros((3, 0), device=device)))
        #     continue
        #
        # # Predictions
        # pred[:, 5] = 0  # if single_cls
        # predn = pred.clone()
        # scale_coords(Img.shape[1:], predn[:, :4], shape, org_shape[1])  # native-space pred
        #
        # # Evaluate
        # if nl:
        #     tbox = xywh2xyxy(labels[:, 1:5])  # target boxes
        #     scale_coords(Img.shape[1:], tbox, shape, org_shape[1])  # native-space labels
        #     labelsn = torch.cat((labels[:, 0:1], tbox), 1)  # native-space labels
        #     correct = process_batch(predn, labelsn, iou_thres)  # 根据预测框和标注框计算correct
        #     if plots:
        #         confusion_matrix.process_batch(predn, labelsn)  # 计算confusion_matrix
        # stats.append((correct, pred[:, 4], pred[:, 5], labels[:, 0]))  # (correct, conf, pcls, tcls)

    # # Compute metrics
    # stats = [torch.cat(x, 0).cpu().numpy() for x in zip(*stats)]  # to numpy
    # if len(stats) and stats[0].any():
    #     # by zk: add thresh_bestF1, thresh_bestR
    #     tp, fp, p, r, f1, ap, ap_class, thresh_bestF1, thresh_bestR = ap_per_class(*stats, plot=plots,
    #                                                                                save_dir=save_dir, names=names)
    #     logging.info('conf_thres_bestF1:{} / conf_thres_bestR:{}'.format(thresh_bestF1, thresh_bestR))
    #     ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
    #     mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
    #     nt = np.bincount(stats[3].astype(np.int64), minlength=nc)  # number of targets per class
    # else:
    #     nt = torch.zeros(1)
    #
    # # Print results
    # pf = '%20s' + '%11i' * 2 + '%11.3g' * 4  # print format
    # logging.info(pf % ('all', seen, nt.sum(), mp, mr, map50, map))

    # logging
    # print('mean/min conf of test_class{}: {}/{}'.
    #       format(test_class, np.mean(pred_conf_class), np.min(pred_conf_class)))
    # print('mean/min conf of nonclass: {}/{}'.format(np.mean(pred_conf_nonclass), np.min(pred_conf_nonclass)))

    logging.info(f'conf_thres:{conf_thres}, iou_thres:{iou_thres}')
    if check_labels:
        number_missing = len(missing_label_list)
        number_error = len(error_label_list)
        logging.info(f'带标签图像数：{number_images_labeled}，标签数量:{number_total_labels}\n')
        logging.info(f'误检/漏标{number_missing}个:{missing_label_list}\n')
        logging.info(f'漏检/误标{number_error}个:{error_label_list}\n')
    if detect_Nonclass:
        logging.info('total/class/nonclass images: {}/{}/{}'.
                     format(len(img_paths), len(class_image_list), len(img_paths) - len(class_image_list)))
        logging.info('preds: {}, min conf:{} on images with labels'.
                     format(len(pred_conf_class), np.min(pred_conf_class)))
        logging.info('preds: {} on images without labels'.
                     format(len(pred_conf_nonclass)))
        if len(pred_conf_nonclass) > 0:
            logging.info('max conf:{} on images without labels'.format(np.max(pred_conf_nonclass)))


if __name__ == "__main__":
    main()
