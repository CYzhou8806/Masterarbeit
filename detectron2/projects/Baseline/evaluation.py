#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@Project ：Masterarbeit
@File    ：evaluation.py
@Author  ：Yu Cao
@Date    ：2022/1/24 13:49
"""
import os
from PIL import Image
import numpy as np
from sklearn import metrics
from tqdm import tqdm
from prettytable import PrettyTable


def pixel_error_rate(prediction, ground_truth, threshold=5):
    '''
    Calculate the pixel_error_rate between prediction and ground truth
    Args:
        prediction: the prediction of the network
        ground_truth:
        threshold: 1 pixel or 3 pixel or 5 pixel

    Returns:

    '''
    assert prediction.shape == ground_truth.shape
    if prediction.shape[0] == 0:
        return None
    count = .0
    for i in range(prediction.shape[0]):
        # for j in range(prediction.shape[1]):
        if abs(prediction[i] - ground_truth[i]) > threshold:
            count += 1.0
    return count / float(prediction.shape[0])


def eval_disparity(predict_root, gt_root, output=None):
    '''
    Make evaluation of the disparity prediction.
    It contains 'MAE', 'RMSE', 'PER_1', 'PER_3', 'PER_5'
    Args:
        predict_root: the root folder of prediction
        gt_root: the root folder of ground truth

    Returns:
        the results will be saved in a new txt file "disparity_evaluation.txt"

    '''
    detail_res = {}
    list_PER_5 = []
    list_PER_3 = []
    list_PER_1 = []
    list_RMSE = []
    list_MAE = []
    list_files = []

    for root, dirs, files in os.walk(predict_root):
        for file in tqdm(files):
            if os.path.splitext(file)[-1] == '.png':
                # load prediction and ground truth
                gt_file = os.path.join(gt_root, file)
                predict_file = os.path.join(root, file)
                gt = np.asarray(Image.open(gt_file)) / 256
                predict = np.asarray(Image.open(predict_file)) / 256

                # only eval the pixel with label
                mask = (gt != 0)
                predict_mask = predict[mask]
                gt_mask = gt[mask]

                PER_5 = pixel_error_rate(predict_mask, gt_mask, threshold=5)
                PER_3 = pixel_error_rate(predict_mask, gt_mask, threshold=3)
                PER_1 = pixel_error_rate(predict_mask, gt_mask, threshold=1)
                RMSE = metrics.mean_squared_error(predict_mask, gt_mask) ** 0.5
                MAE = metrics.mean_absolute_error(predict_mask, gt_mask)
                detail_res[os.path.splitext(file)[0]] = [MAE, RMSE, PER_1, PER_3, PER_5]
                list_PER_5.append(PER_5)
                list_PER_3.append(PER_3)
                list_PER_1.append(PER_1)
                list_RMSE.append(RMSE)
                list_MAE.append(MAE)
                list_files.append(os.path.splitext(file)[0])

    # create details table
    tabel_detail = PrettyTable(['File', 'MAE', 'RMSE', 'PER_1', 'PER_3', 'PER_5'])
    for k, v in detail_res.items():
        tabel_detail.add_row(
            [k, str(format(v[0], '.5f')), str(format(v[1], '.5f')), str(format(v[2], '.5f')), str(format(v[3], '.5f')),
             str(format(v[4], '.5f'))])
    print(tabel_detail)
    print('\n\n')

    if output:
        # dis_eval_result = os.path.join(output, 'disparity_evaluation.txt')
        dis_eval_result = output
    else:
        dis_eval_result = "disparity_evaluation.txt"
    f = open(dis_eval_result, "w")
    tabel_detail_txt = tabel_detail.get_string()
    f.write(tabel_detail_txt)
    f.write('\n\n\n')

    # create summery table
    for i, l in enumerate([list_MAE, list_RMSE, list_PER_1, list_PER_3, list_PER_5]):
        max_value = max(l)
        min_value = min(l)
        max_file = list_files[l.index(max_value)]
        min_file = list_files[l.index(min_value)]
        average_value = np.mean(l)

        if i == 0:
            tabel_summary = PrettyTable(['Average_MAE', 'Max_MAE', 'Max_MAE_file', 'Min_MAE', 'Min_MAE_file'])
        elif i == 1:
            tabel_summary = PrettyTable(['Average_RMSE', 'Max_RMSE', 'Max_RMSE_file', 'Min_RMSE', 'Min_RMSE_file'])
        elif i == 2:
            tabel_summary = PrettyTable(['Average_PER_1', 'Max_PER_1', 'Max_PER_1_file', 'Min_PER_1', 'Min_PER_1_file'])
        elif i == 3:
            tabel_summary = PrettyTable(['Average_PER_3', 'Max_PER_3', 'Max_PER_3_file', 'Min_PER_3', 'Min_PER_3_file'])
        else:
            tabel_summary = PrettyTable(['Average_PER_5', 'Max_PER_5', 'Max_PER_5_file', 'Min_PER_5', 'Min_PER_5_file'])
        tabel_summary.add_row(
            [str(format(average_value, '.5f')), str(format(max_value, '.5f')), max_file, str(format(min_value, '.5f')),
             min_file])
        print(tabel_summary)
        print('\n')

        tabel_summary_txt = tabel_summary.get_string()
        f.write(tabel_summary_txt)
        f.write('\n\n')

    f.close()
    return dis_eval_result


if __name__ == "__main__":
    dis_prediction_dir = r"C:\Users\cyzho\Desktop\demo_test_new\prediction_kitti2015_test_03"
    dis_ground_truth_dir = r"C:\Users\cyzho\Desktop\demo_test_new\disp_occ_0_test"

    eval_disparity(dis_prediction_dir, dis_ground_truth_dir)
