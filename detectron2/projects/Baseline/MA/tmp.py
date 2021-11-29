#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@Project ：Masterarbeit
@File    ：tmp.py
@Author  ：Yu Cao
@Date    ：2021/11/29 17:48 
"""
import time

import numpy as np
import torch
import cv2

from detectron2.data import detection_utils as utils

count = 0
disparity_file_name = r"C:\Users\cyzho\Desktop\LUIS\berlin_000000_000019_disparity.png"
disparity_gt = utils.read_image(disparity_file_name, "RGB")
disparity_gt = disparity_gt[:, :, 0]
disparity_gt = disparity_gt.astype(float)

start = time.time()
for i in range(disparity_gt.shape[0]):
    for j in range(disparity_gt.shape[1]):
        if disparity_gt[i, j] > 0:
            count = count+1
            disparity_gt[i, j] = (disparity_gt[i, j] - 1.) / 256
end = time.time()
print (end-start)
disparity_gt_torch = torch.as_tensor(np.ascontiguousarray(disparity_gt, dtype=np.float32))
print(count)


disparity_gt_other = utils.read_image(disparity_file_name, "RGB")
disparity_gt_other = disparity_gt_other[:, :, 0]
disparity_gt_other = disparity_gt_other.astype(float)
start = time.time()
mask = disparity_gt_other > 0.0
disparity_gt_other[mask] = (disparity_gt_other[mask] - 1.) / 256
end = time.time()
print (end-start)

print(np.all(disparity_gt_other == disparity_gt))

