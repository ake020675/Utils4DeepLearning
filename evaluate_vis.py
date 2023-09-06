import os
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import shutil


def dice(y_true, y_pred):
    return (2 * (y_true * y_pred).sum() + 1e-15) / (y_true.sum() + y_pred.sum() + 1e-15)


def general_dice(y_true, y_pred):
    if y_true.sum() == 0:
        if y_pred.sum() == 0:
            return 1
        else:
            return 0

    return dice(y_true, y_pred)


def jaccard(y_true, y_pred):
    intersection = (y_true * y_pred).sum()
    union = y_true.sum() + y_pred.sum() - intersection
    return (intersection + 1e-15) / (union + 1e-15)


def general_jaccard(y_true, y_pred):
    if y_true.sum() == 0:
        if y_pred.sum() == 0:
            return 1
        else:
            return 0

    return jaccard(y_true, y_pred)


def evaluate(
        ground_truth_dir='../SA-UNet-master/CHASE/test_ps/label',
        pred_dir=r'../SA-UNet-master/CHASE/test_ps/result',
        threshold=0.5,
        input_size=None,
        gt_ext='jpg',
        pred_ext='png'
):
    result_dice = []
    result_jaccard = []

    gt_paths = [path for path in Path(ground_truth_dir).glob('*')]
    pred_paths = [path for path in Path(pred_dir).glob('*')]

    # gt_mask_list = np.zeros((len(gt_paths), input_size, input_size), np.uint8)  # by zk
    # pred_mask_list = np.zeros((len(gt_paths), input_size, input_size), np.uint8)  # by zk

    for i, gt_file in enumerate(tqdm(gt_paths)):

        y_true = (cv2.imread(str(gt_file), 0) > 0).astype(np.uint8)
        # label_image = cv2.resize(cv2.imread(str(gt_file), 0), (args.input_size, args.input_size))  # add resize by zk
        # cv2.normalize(label_image, label_image, norm_type=cv2.NORM_MINMAX)
        # label_image = (label_image * 255).astype(np.uint8)
        # y_true = label_image > 0

        pred_file_name = Path(pred_dir) / gt_file.name
        if not pred_file_name.exists():
            pred_file_name = Path(pred_dir) / gt_file.name.replace(gt_ext, pred_ext)

        if not pred_file_name.exists():
            print(f'missing prediction for file {gt_file.name}')
            continue

        pred_image = cv2.imread(str(pred_file_name), 0)
        if input_size is not None:
            pred_image = cv2.resize(pred_image, (input_size, input_size))  # add resize by zk
        y_pred = (pred_image > 255 * threshold).astype(np.uint8)

        result_dice += [dice(y_true, y_pred)]
        result_jaccard += [jaccard(y_true, y_pred)]

        # pred_mask_list[i, :, :] = pred_image / 255.0  # by zk
        # gt_mask_list[i, :, :] = y_true  # by zk

    return result_dice, result_jaccard


# visualize gt + pred on 1 image
def export_visualize(save_dir, pred_path, gt_dir, gt_ext='jpg'):
    # get img
    name = pred_path.stem
    gt_path = os.path.join(gt_dir, name + '.' + gt_ext)

    gt = cv2.imread(str(gt_path), 0)
    pred = cv2.imread(str(pred_path), 0)

    # plot
    fig_img, ax_img = plt.subplots(1, 2, figsize=(8 * 1 / 2.54, 16 * 1 / 2.54))
    for ax_i in ax_img:
        ax_i.axes.xaxis.set_visible(False)
        ax_i.axes.yaxis.set_visible(False)
        ax_i.spines['top'].set_visible(False)
        ax_i.spines['right'].set_visible(False)
        ax_i.spines['bottom'].set_visible(False)
        ax_i.spines['left'].set_visible(False)
    plt.subplots_adjust(hspace=0.1, wspace=0.1)
    ax_img[0].set_title('pred', loc='center')
    ax_img[1].set_title('gt', loc='center')
    ax_img[0].imshow(pred, cmap='gray', interpolation='none')
    ax_img[1].imshow(gt, cmap='gray', interpolation='none')

    save_name = os.path.join(save_dir, name + '.png')
    fig_img.savefig(save_name, dpi=500, format='png', bbox_inches='tight', pad_inches=0.0)  # by zk

    plt.clf()  # 清图。
    plt.cla()  # 清坐标轴。
    plt.close()  # 关窗口
    # plt.close('all')


if __name__ == '__main__':

    # args
    # crack
    # models = ["SA-UNet", "TransUNet", "unetPlus2", "unet-VGG16"]
    # colors = ['r', 'g', 'b', 'y']
    # ground_truth_dir = '../SA-UNet-master/CHASE/test_ps/label'
    # pred_ext = 'png',
    # gt_ext = 'jpg'

    # leakage models
    colors = ['r', 'g']

    # models = ["TransUNet", "efficientUNet++"]
    models = ["TransUNet"]  # leakage models
    ground_truth_dir = r'C:\Users\zk\Desktop\TransUNet-main\leakage\val\mask'
    pred_dir = r'C:\Users\zk\Desktop\TransUNet-main\predictions\leakage'
    path_orignImage = r'D:\火眼智能\隧道病害分析\渗水\标注数据\leakage-SEG\images'

    # ground_truth_dir = r'C:\Users\zk\Desktop\Project-Segmentation\EfficientUNetPlusPlus-main\leakage\test\mask'
    # pred_dir = r'C:\Users\zk\Desktop\Project-Segmentation\EfficientUNetPlusPlus-main\leakage\test\pred'

    gt_ext = 'png'
    pred_ext = 'png'
    result_dir = 'result/'

    # compute and visualize scores
    fig_dice, ax_dice = plt.subplots()
    fig_jaccard, ax_jaccard = plt.subplots()
    for i, model in enumerate(models):

        result_dice, result_jaccard = evaluate(ground_truth_dir, pred_dir, gt_ext=gt_ext, pred_ext=pred_ext)
        im_paths = [path for path in Path(pred_dir).glob('*')]

        print(f'stats of {model}: mean, std')
        print('Dice = ', np.mean(result_dice), np.std(result_dice))  # f1 score
        print('Jaccard = ', np.mean(result_jaccard), np.std(result_jaccard))  # mIOU

        # plot dice and jaccard - by zk
        # Data for plotting
        n = np.arange(0.0, len(result_dice), 1.0)
        ax_dice.plot(n, result_dice, color=colors[i], label=model)
        ax_jaccard.plot(n, result_jaccard, color=colors[i], label=model)

        # good/bad samples
        total_score = np.sum((result_dice, result_jaccard), 0)/2
        indexs = np.argsort(total_score)
        indexs_bottom = indexs[:10]
        indexs_top = indexs[-11:-1]
        good_dir = result_dir + 'good/' + model
        if not os.path.exists(good_dir):
            os.makedirs(good_dir)
        bad_dir = result_dir + 'bad/' + model
        if not os.path.exists(bad_dir):
            os.makedirs(bad_dir)

        # visualize pred and gt
        # for j in range(10):
        for j in range(len(total_score)):
            score = total_score[j]

            # good_ind = indexs_top[j]
            # good_pred_path = im_paths[good_ind]
            # export_visualize(good_dir, good_pred_path, ground_truth_dir, gt_ext='png')

            if score < 0.7:
                bad_pred_path = im_paths[j]
                export_visualize(bad_dir, bad_pred_path, ground_truth_dir, gt_ext='png')
                name = os.path.basename(bad_pred_path).split('.')[0]
                path_origin_image = os.path.join(path_orignImage, name + '.jpg')
                shutil.copy(path_origin_image, bad_dir)

    # save and show plot of dice
    ax_dice.legend(title='Models')
    ax_dice.set(xlabel='image', ylabel='dice',
                title='dice(average F1 score per image)')
    ax_dice.grid()
    fig_dice.savefig(result_dir + "dice.png")
    plt.show()

    # save and show plot of jaccard
    ax_jaccard.legend(title='Models')
    ax_jaccard.set(xlabel='image', ylabel='jaccard',
                   title='jaccard(IOU per image)')
    ax_jaccard.grid()
    fig_jaccard.savefig(result_dir + "jaccard.png")
    plt.show()

    ########## compute best seg threshold - by zk ##########
    # from sklearn.metrics import roc_auc_score, auc, precision_recall_curve
    #
    # # gt_mask = np.squeeze(np.asarray(gt_mask_list, dtype=np.bool), axis=1)
    # # pred_mask = np.squeeze(np.asarray(pred_mask_list, dtype=np.float16), axis=1)
    # gt_mask = np.asarray(gt_mask_list, dtype=np.bool)
    # pred_mask = np.asarray(pred_mask_list, dtype=np.float16)
    #
    # # debug_gt_mask = gt_mask.flatten()
    # # debug_pred_mask = pred_mask.flatten()
    #
    # seg_auc = roc_auc_score(gt_mask.flatten(), pred_mask.flatten())
    # precision, recall, thresholds = precision_recall_curve(gt_mask.flatten(), pred_mask.flatten())
    # a = 2 * precision * recall
    # b = precision + recall
    # f1 = np.divide(a, b, out=np.zeros_like(a), where=b != 0)
    # seg_threshold = thresholds[np.argmax(f1)]  # 分割阈值选取: 令f1=a/b最大的thresholds
    # print('Optimal SEG Threshold: {:.2f}'.format(seg_threshold))
    # print('Optimal SEG AUC: {:.2f}'.format(seg_auc))
    # print('Optimal SEG f1: {:.2f}'.format(np.max(f1)))
    ########## compute best seg threshold - by zk ##########
