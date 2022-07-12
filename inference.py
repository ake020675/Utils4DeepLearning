"""
by zk
20210105
solo模型推断
输入：图像/文件夹路径
输出：类别labels、目标框bbox、分割mask
"""

from mmdet.apis import init_detector, inference_detector
import mmcv
# from mmdet.core.visualization.image import imshow_det_bboxes
import numpy as np
import torch
import os
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from PIL import Image
import cv2

EPS = 1e-2


def color_val_matplotlib(color):
    """Convert various input in BGR order to normalized RGB matplotlib color
    tuples,

    Args:
        color (:obj:`Color`/str/tuple/int/ndarray): Color inputs

    Returns:
        tuple[float]: A tuple of 3 normalized floats indicating RGB channels.
    """
    color = mmcv.color_val(color)
    color = [color / 255 for color in color[::-1]]
    return tuple(color)


def imshow_det_bboxes(img,
                      bboxes,
                      labels,
                      segms=None,
                      class_names=None,
                      score_thr=0,
                      bbox_color='green',
                      text_color='green',
                      mask_color=None,
                      thickness=2,
                      font_size=13,
                      win_name='',
                      show=True,
                      wait_time=0,
                      out_file=None):
    """Draw bboxes and class labels (with scores) on an image.

    Args:
        img (str or ndarray): The image to be displayed.
        bboxes (ndarray): Bounding boxes (with scores), shaped (n, 4) or
            (n, 5).
        labels (ndarray): Labels of bboxes.
        segms (ndarray or None): Masks, shaped (n,h,w) or None
        class_names (list[str]): Names of each classes.
        score_thr (float): Minimum score of bboxes to be shown.  Default: 0
        bbox_color (str or tuple(int) or :obj:`Color`):Color of bbox lines.
           The tuple of color should be in BGR order. Default: 'green'
        text_color (str or tuple(int) or :obj:`Color`):Color of texts.
           The tuple of color should be in BGR order. Default: 'green'
        mask_color (str or tuple(int) or :obj:`Color`, optional):
           Color of masks. The tuple of color should be in BGR order.
           Default: None
        thickness (int): Thickness of lines. Default: 2
        font_size (int): Font size of texts. Default: 13
        show (bool): Whether to show the image. Default: True
        win_name (str): The window name. Default: ''
        wait_time (float): Value of waitKey param. Default: 0.
        out_file (str, optional): The filename to write the image.
            Default: None

    Returns:
        ndarray: The image with bboxes drawn on it.
    """
    assert bboxes.ndim == 2, \
        f' bboxes ndim should be 2, but its ndim is {bboxes.ndim}.'
    assert labels.ndim == 1, \
        f' labels ndim should be 1, but its ndim is {labels.ndim}.'
    assert bboxes.shape[0] == labels.shape[0], \
        'bboxes.shape[0] and labels.shape[0] should have the same length.'
    assert bboxes.shape[1] == 4 or bboxes.shape[1] == 5, \
        f' bboxes.shape[1] should be 4 or 5, but its {bboxes.shape[1]}.'
    img = mmcv.imread(img).astype(np.uint8)

    if score_thr > 0:
        assert bboxes.shape[1] == 5
        scores = bboxes[:, -1]  # 每个bbox的最后一个值是分数
        inds = scores > score_thr  # 大于阈值的bbox序号
        bboxes = bboxes[inds, :]  # 大于阈值的bbox
        labels = labels[inds]  # 大于阈值的bbox对应的label
        confs = scores[inds]
        if segms is not None:
            segms = segms[inds, ...]  # 大于阈值的bbox对应的mask

    mask_colors = []
    if labels.shape[0] > 0:
        if mask_color is None:  # 未指定颜色，随机分配
            # Get random state before set seed, and restore random state later.
            # Prevent loss of randomness.
            # See: https://github.com/open-mmlab/mmdetection/issues/5844
            state = np.random.get_state()
            # random color
            np.random.seed(42)
            mask_colors = [
                np.random.randint(0, 256, (1, 3), dtype=np.uint8)
                for _ in range(max(labels) + 1)
            ]
            np.random.set_state(state)
        else:  # 若指定了颜色，按序号分配
            # specify  color
            mask_colors = [
                np.array(mmcv.color_val(mask_color)[::-1], dtype=np.uint8)
            ] * (
                max(labels) + 1)

    bbox_color = color_val_matplotlib(bbox_color)
    text_color = color_val_matplotlib(text_color)

    img = mmcv.bgr2rgb(img)
    width, height = img.shape[1], img.shape[0]
    img = np.ascontiguousarray(img)

    # fig = plt.figure(win_name, frameon=False)
    # plt.title(win_name)
    # canvas = fig.canvas
    # dpi = fig.get_dpi()
    # # add a small EPS to avoid precision lost due to matplotlib's truncation
    # # (https://github.com/matplotlib/matplotlib/issues/15363)
    # fig.set_size_inches((width + EPS) / dpi, (height + EPS) / dpi)
    #
    # # remove white edges by set subplot margin
    # plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
    # ax = plt.gca()
    # ax.axis('off')

    polygons = []
    color = []
    for i, (bbox, label) in enumerate(zip(bboxes, labels)):
        bbox_int = bbox.astype(np.int32)
        poly = [[bbox_int[0], bbox_int[1]], [bbox_int[0], bbox_int[3]],
                [bbox_int[2], bbox_int[3]], [bbox_int[2], bbox_int[1]]]
        np_poly = np.array(poly).reshape((4, 2))
        polygons.append(Polygon(np_poly))
        color.append(bbox_color)
        label_text = class_names[
            label] if class_names is not None else f'class {label}'
        if len(bbox) > 4:
            label_text += f'|{bbox[-1]:.02f}'
        # ax.text(
        #     bbox_int[0],
        #     bbox_int[1],
        #     f'{label_text}',
        #     bbox={
        #         'facecolor': 'black',
        #         'alpha': 0.8,
        #         'pad': 0.7,
        #         'edgecolor': 'none'
        #     },
        #     color=text_color,
        #     fontsize=font_size,
        #     verticalalignment='top',
        #     horizontalalignment='left')
        if segms is not None:
            color_mask = mask_colors[labels[i]]
            mask = segms[i].astype(bool)
            img[mask] = img[mask] * 0.5 + color_mask * 0.5  # mask对应位置着色、覆盖原图


            # # debug by zk:
            # winname = 'mask_' + str(i)
            # cv2.namedWindow(winname)
            # cv2.imshow(winname, segms[i].astype(np.uint8)*255)
            # cv2.waitKey()
            # cv2.destroyAllWindows()

    # plt.imshow(segms[0, :].astype(np.uint8)*255)
    # plt.imshow(img)

    p = PatchCollection(
        polygons, facecolor='none', edgecolors=color, linewidths=thickness)
    # ax.add_collection(p)

    # stream, _ = canvas.print_to_buffer()
    # buffer = np.frombuffer(stream, dtype='uint8')
    # img_rgba = buffer.reshape(height, width, 4)
    # rgb, alpha = np.split(img_rgba, [3], axis=2)
    # img = rgb.astype('uint8')
    img = mmcv.rgb2bgr(img)

    # if show:
    #     # We do not use cv2 for display because in some cases, opencv will
    #     # conflict with Qt, it will output a warning: Current thread
    #     # is not the object's thread. You can refer to
    #     # https://github.com/opencv/opencv-python/issues/46 for details
    #     if wait_time == 0:
    #         plt.show()
    #     else:
    #         plt.show(block=False)
    #         plt.pause(wait_time)
    if out_file is not None:
        # mmcv.imwrite(img, out_file)
        Image.fromarray(img).save(out_file)
    # plt.close()

    # return img


def show_result(
                # self,
                CLASSES,  # BY ZK
                img,
                result,
                score_thr=0.3,
                bbox_color=(72, 101, 241),
                text_color=(72, 101, 241),
                mask_color=None,
                thickness=2,
                font_size=13,
                win_name='',
                show=False,
                wait_time=0,
                out_file=None):
    """Draw `result` over `img`.

    Args:
        img (str or Tensor): The image to be displayed.
        result (tuple): Format bbox and mask results.
            It contains two items:

            - bbox_results (list[np.ndarray]): BBox results of
              single image. The list corresponds to each class.
              each ndarray has a shape (N, 5), N is the number of
              bboxes with this category, and last dimension
              5 arrange as (x1, y1, x2, y2, scores).
            - mask_results (list[np.ndarray]): Mask results of
              single image. The list corresponds to each class.
              each ndarray has shape (N, img_h, img_w), N
              is the number of masks with this category.

        score_thr (float, optional): Minimum score of bboxes to be shown.
            Default: 0.3.
        bbox_color (str or tuple(int) or :obj:`Color`):Color of bbox lines.
           The tuple of color should be in BGR order. Default: 'green'
        text_color (str or tuple(int) or :obj:`Color`):Color of texts.
           The tuple of color should be in BGR order. Default: 'green'
        mask_color (None or str or tuple(int) or :obj:`Color`):
           Color of masks. The tuple of color should be in BGR order.
           Default: None
        thickness (int): Thickness of lines. Default: 2
        font_size (int): Font size of texts. Default: 13
        win_name (str): The window name. Default: ''
        wait_time (float): Value of waitKey param.
            Default: 0.
        show (bool): Whether to show the image.
            Default: False.
        out_file (str or None): The filename to write the image.
            Default: None.

    Returns:
        img (Tensor): Only if not `show` or `out_file`
    """

    assert isinstance(result, tuple)
    bbox_result, mask_result = result
    bboxes = np.vstack(bbox_result)
    img = mmcv.imread(img)
    img = img.copy()
    labels = [
        np.full(bbox.shape[0], i, dtype=np.int32)
        for i, bbox in enumerate(bbox_result)
    ]
    labels = np.concatenate(labels)
    if len(labels) == 0:
        bboxes = np.zeros([0, 5])
        masks = np.zeros([0, 0, 0])
    # draw segmentation masks
    else:
        masks = mmcv.concat_list(mask_result)

        # debug by zk:
        # for i in range(len(masks)):
        #     winname = 'mask_' + str(i)
        #     cv2.namedWindow(winname)
        #     cv2.imshow(winname, masks[i].astype(np.uint8)*255)
        #     cv2.waitKey()
        #     cv2.destroyAllWindows()

        if isinstance(masks[0], torch.Tensor):
            masks = torch.stack(masks, dim=0).detach().cpu().numpy()
        else:
            masks = np.stack(masks, axis=0)
        # dummy bboxes
        if bboxes[:, :4].sum() == 0:  # 存放在[:, 0:3]维中的bbox信息，都是0
            num_masks = len(bboxes)
            x_any = masks.any(axis=1)
            y_any = masks.any(axis=2)
            for idx in range(num_masks):
                x = np.where(x_any[idx, :])[0]
                y = np.where(y_any[idx, :])[0]
                if len(x) > 0 and len(y) > 0:
                    bboxes[idx, :4] = np.array(
                        [x[0], y[0], x[-1] + 1, y[-1] + 1],
                        dtype=np.float32)  # bbox信息，存放在[idx, 0:3]维
    # if out_file specified, do not show image in window
    if out_file is not None:
        show = False

    # draw bounding boxes
    imshow_det_bboxes(
        img,
        bboxes,
        labels,
        masks,
        # class_names=self.CLASSES,
        class_names=CLASSES,  # BY ZK

        score_thr=score_thr,
        bbox_color=bbox_color,
        text_color=text_color,
        mask_color=mask_color,
        thickness=thickness,
        font_size=font_size,
        win_name=win_name,
        show=show,
        wait_time=wait_time,
        out_file=out_file)

    # by zk : postprocess in
    if score_thr > 0:
        assert bboxes.shape[1] == 5
        scores = bboxes[:, -1]  # 每个bbox的最后一个值是分数
        inds = scores > score_thr  # 大于阈值的bbox序号
        bboxes = bboxes[inds, :]  # 大于阈值的bbox
        labels = labels[inds]  # 大于阈值的bbox对应的label
        confs = scores[inds]
        if masks is not None:
            segms = masks[inds, ...]  # 大于阈值的bbox对应的mask

    # if not (show or out_file):
    return bboxes, labels, confs, segms


def get_sal(img, sal_path):
    filename = os.path.basename(img)
    sal_impath = os.path.join(sal_path, filename)
    sal = Image.open(sal_impath).convert('L')
    return np.array(sal).astype(np.uint8)


def combine_sal_instance(sal_mask, bboxes, labels, confs, masks, class_names, ifshow=0):

    if ifshow:
        cv2.namedWindow('sal_mask')
        cv2.imshow('sal_mask', sal_mask)
        # cv2.waitKey()

    final_mask = sal_mask
    for i in range(len(masks)):  # 针对各类别mask
        obj_mask = masks[i].astype(np.uint8) * 255
        obj_label = labels[i]
        class_name = class_names[obj_label]

        # debug
        if ifshow:
            cv2.namedWindow(class_name)
            cv2.imshow(class_name, obj_mask)
            cv2.waitKey()

        intersec = cv2.bitwise_and(obj_mask, sal_mask)
        if cv2.countNonZero(intersec) == 0:  # 该类物体与显著图无交集
            continue

        # combine obj_mask and sal_mask
        # final_mask = np.zeros_like(obj_mask)
        contours_sal, _ = cv2.findContours(sal_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        contours_obj, _ = cv2.findContours(obj_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        # 一. 剔除正常部件分割中的错误结果
        selected_contours_obj = list()
        list_contours_obj = list()
        if class_name == "05":  # 1.位置原则：若部件类别为05，筛除偏离图像中心的轮廓
            for i in range(len(contours_obj)):
                h, w = obj_mask.shape[:2]
                bR = cv2.boundingRect(contours_obj[i])
                center_x = bR[0] + bR[2] / 2
                if center_x > 0.6 * w or center_x < 0.4 * w:
                    # list_contours_obj.remove(list_contours_obj[i])
                    continue
                else:
                    list_contours_obj.append(contours_obj[i])
        else:
            list_contours_obj = list(contours_obj)

        # if class_name != "04":
        if len(list_contours_obj) > 1:
            max_area = 0
            for i in range(len(list_contours_obj)):  # 2.面积原则：轮廓数大于1，筛除面积较小的区域
                area = cv2.contourArea(list_contours_obj[i])
                if area > max_area:
                    max_area = area
            for i in range(len(list_contours_obj)):
                area = cv2.contourArea(list_contours_obj[i])
                if area > 0.6 * max_area:
                    selected_contours_obj.append(list_contours_obj[i])
                # if area < 0.6 * max_area:
                #     list_contours_obj.remove(list_contours_obj[i])

        if len(selected_contours_obj) == 0:  # 保留所有区域
            selected_contours_obj = list_contours_obj

        # debug step 1
        if ifshow:
            class_mask = np.zeros_like(obj_mask)  # 类别i中的当前物体
            cv2.drawContours(class_mask, selected_contours_obj, -1, 255, -1)
            cv2.namedWindow('class_mask')
            cv2.imshow('class_mask', class_mask)
            cv2.waitKey()

        # 二. 剔除显著图中的正常部件
        if selected_contours_obj is None:
            continue

        for contour_obj in selected_contours_obj:
            # 面积太小的物体
            if cv2.contourArea(contour_obj) < 16:
                continue

            cur_obj_mask = np.zeros_like(obj_mask)  # 类别i中的当前物体
            cv2.drawContours(cur_obj_mask, [contour_obj], 0, 255, -1)

            # 当前物体与显著图无交集：不处理
            num_intersect = cv2.countNonZero(cv2.bitwise_and(cur_obj_mask, sal_mask))
            if num_intersect == 0:
                continue
            elif len(contours_sal) == 1 and num_intersect < 0.5 * cv2.contourArea(contours_sal[0]):
                continue
            else:
                # 当前物体与显著图有交集
                bR = cv2.boundingRect(contour_obj)
                final_mask[bR[1]:bR[1] + bR[3], bR[0]:bR[0] + bR[2]] = 0

            # for contour_sal in contours_sal:
            #     sal_obj = np.zeros_like(sal_mask)
            #     cv2.drawContours(sal_obj, [contour_sal], 0, 255, -1)
            #     sal_area = cv2.countNonZero(sal_obj)
            #
            #     num_intersect = cv2.countNonZero(cv2.bitwise_and(cur_obj_mask, sal_obj))
            #     if num_intersect > sal_area / 2:  # 正常部件和显著区域有交叉， 显著区域置为0
            #
            #         # # debug
            #         # cv2.namedWindow('cur_obj_mask')
            #         # cv2.imshow('cur_obj_mask', cur_obj_mask)
            #         # cv2.namedWindow('sal_obj')
            #         # cv2.imshow('sal_obj', sal_obj)
            #         # cv2.waitKey()
            #
            #         # draw_contours.append(contour_sal)
            #         bR = cv2.boundingRect(contour_sal)
            #         final_mask[bR[1]:bR[1] + bR[3], bR[0]:bR[0] + bR[2]] = 0

    # 显示最终结果
    if ifshow:
        cv2.namedWindow('final_mask')
        cv2.imshow('final_mask', final_mask)
        cv2.waitKey()
    cv2.destroyAllWindows()

    return final_mask


if __name__ == '__main__':

    # Specify the path to model config and checkpoint file
    config_file = r'C:\Users\zk\Desktop\mmdetection-2.19.1\configs\solo\decoupled_solo_light_r50_fpn_3x_TunnelNormal.py'
    checkpoint_file = 'tools/work_dirs/decoupled_solo_light_r50_fpn_3x_TunnelNormal/epoch_36.pth'
    sal_path = r'C:\Users\zk\Desktop\Project-anomaly detection\cflow-ad-master\saliency_AD\salmap\test\anomal'

    # build the model from a config file and a checkpoint file
    model = init_detector(config_file, checkpoint_file, device='cuda:0')

    # test a single image and show the results
    img = 'TunnelNormal_coco/anomal/2243.jpg'  # todo: debug image:
    result = inference_detector(model, img)  # bbox_result, mask_result
    # visualize the results in a new window
    # model.show_result(img, result)
    # or save the visualization results to image files
    # model.show_result(img, result, out_file='result.jpg', score_thr=0.1)
    # BY ZK
    # show_result(model.CLASSES, img, result, out_file='result.jpg', score_thr=0.6)
    bboxes, labels, confs, masks = show_result(model.CLASSES, img, result, score_thr=0.8,  out_file='result.jpg')  # BY ZK
    # conbine with sal map
    sal_mask = get_sal(img, sal_path)
    if labels is None:
        final_mask = sal_mask
    else:
        final_mask = combine_sal_instance(sal_mask, bboxes, labels, confs, masks, class_names=model.CLASSES, ifshow=1)

    # # test multiple images and save results
    # test_path = 'TunnelNormal_coco/anomal'
    # save_path = test_path + '_result'
    # if not os.path.exists(save_path):
    #     os.makedirs(save_path)
    # viz_path = test_path + '_viz'
    # if not os.path.exists(viz_path):
    #     os.makedirs(viz_path)
    #
    # for file in os.listdir(test_path):
    #     img = os.path.join(test_path, file)
    #     print('inferencing {}'.format(img))
    #     save_name = os.path.join(save_path, file)
    #
    #     result = inference_detector(model, img)
    #     # model.show_result(img, result, out_file=test_path+'_solo/'+file, score_thr=0.6)
    #     # BY ZK
    #     # show_result(model.CLASSES, img, result, score_thr=0.7, out_file=save_name)
    #     bboxes, labels, confs, masks = show_result(model.CLASSES, img, result, score_thr=0.8, out_file=test_path+'_solo/'+file)  # BY ZK
    #     sal_mask = get_sal(img, sal_path)
    #     if labels is None:
    #         final_mask = sal_mask
    #     else:
    #         final_mask = combine_sal_instance(sal_mask, bboxes, labels, confs, masks, class_names=model.CLASSES)
    #     Image.fromarray(final_mask).save(save_name)
    #
    #     # for viz
    #     im = Image.open(img).convert('RGB')
    #     rgb_im = np.array(im, dtype=np.uint8)
    #     viz = rgb_im
    #     inds = final_mask > 32
    #     viz[inds] = 0.8*rgb_im[inds] + [0, 50, 0]
    #     Image.fromarray(viz).save(os.path.join(viz_path, file))
