"""
by zk
2024.4.22
yolov9检测模型推理脚本

get_model: 用于配置模型参数并加载模型的函数；参考predict.py的参数配置方式，主要配置的参数是权重文件、yaml文件、阈值参数。配置后加载模型）
inference: 对单张图进行检测得到置信度最高的一个目标。
"""

import argparse
import os
import platform
import sys
from pathlib import Path

import numpy as np
import torch

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLO root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, scale_segments,
                           strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.segment.general import masks2segments, process_mask
from utils.torch_utils import select_device, smart_inference_mode


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov9-c-converted.pt', help='model path(s)')
    parser.add_argument('--source', type=str, default=ROOT / 'data/images', help='file/dir/URL/glob/screen/0(webcam)')
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='(optional) dataset.yaml path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs/predict-seg', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--vid-stride', type=int, default=1, help='video frame-rate stride')
    parser.add_argument('--retina-masks', action='store_true', help='whether to plot masks in native resolution')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt))
    return opt


def get_model(opt):
    """
    用于配置模型参数并加载模型的函数；参考predict.py的参数配置方式，主要配置的参数是权重文件、yaml文件、阈值参数。配置后加载模型
    """

    # Load model
    device = select_device(opt.device)
    model = DetectMultiBackend(opt.weights, device=device, dnn=opt.dnn, data=opt.data, fp16=opt.half)

    return model


def inference(imgpath, model, conf=0.01, input_size=(640, 640),
              issavedlabeled=False, save_dir=None):
    """
    用于逐张送图并输出检测结果的函数：检测得到置信度最高的一个目标.
    input:
        imgpath：需要输入的单张图片路径
        model: 推理模型
        issavedlabeled：识别后加了box和label名称的结果图片是否存储。如果存储，转存到一个固定地址（这个地址可以手动改就可以）
    output:
        classname：所有检测目标中置信度最大的目标名称（根据class_id到yaml文件中匹配）
        conf：置信度最高的目标的置信度数据（0~1）
    """


    # read image
    Im = cv2.imread(imgpath, 1)
    image_name = os.path.basename(imgpath)

    # preprocess
    height, width = input_size
    input_img = cv2.resize(Im, (width, height))
    img = input_img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    img = np.ascontiguousarray(img)

    # to tensor
    Img = torch.from_numpy(img).to(model.device)
    Img = Img.float()  # uint8 to fp16/32
    Img /= 255  # 0 - 255 to 0.0 - 1.0
    if len(Img.shape) == 3:
        Img = Img[None]  # expand for batch dim

    # inference
    pred, proto = model(Img)[:2]
    # res = model(Img)
    out = pred

    # 非极大值抑制
    pred = non_max_suppression(out, conf_thres=conf, iou_thres=0.1, max_det=100)

    # 缩放box坐标
    for det in pred:  # per image
        if len(det):  # 检测到物体
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_boxes(Img.shape[2:], det[:, :4], Im.shape[:2]).round()

    label = None
    conf = None
    # draw result
    class_names = model.names
    annotator = Annotator(Im, line_width=3, example=str(class_names))
    # Write results
    for *xyxy, conf, index_cls in reversed(det):
        c = int(index_cls)
        label = f'{class_names[c]} {conf:.2f}'
        annotator.box_label(xyxy, label, color=colors(c, True))
    im0 = annotator.result()

    # save result
    if issavedlabeled:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        if save_dir is not None:
            cv2.imwrite(os.path.join(save_dir, image_name), im0)

    return label, conf


def main():
    opt = parse_opt()
    det_model = get_model(opt)
    imgpath = "./data/images/horses.jpg"
    classname, conf = inference(imgpath, det_model,
                                input_size=(640, 640),
                                issavedlabeled=True, save_dir='./')
    print(classname, conf)


if __name__ == "__main__":
    main()
