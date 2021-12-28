#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@Project ：Masterarbeit
@File    ：check_disparity_gt.py
@Author  ：Yu Cao
@Date    ：2021/12/28 10:26 
"""

import re
import numpy as np
import sys
import cv2
import os
from PIL import Image




def to_warped_image(img, disp, direction_str):
    if direction_str == 'r2l':
        direction = - 1
    elif direction_str == 'l2r':
        direction = 1

    map_x = np.zeros((img.shape[0], img.shape[1]), dtype=np.float32)
    map_y = np.zeros((img.shape[0], img.shape[1]), dtype=np.float32)

    for i in range(map_x.shape[0]):
        map_x[i, :] = [x for x in range(map_x.shape[1])]

    if disp.ndim == 3:
        disp = np.squeeze(disp, axis=-1)

    map_x = map_x + disp * direction

    for j in range(map_y.shape[1]):
        map_y[:, j] = [y for y in range(map_y.shape[0])]

    left_warped = cv2.remap(img, map_x, map_y, cv2.INTER_NEAREST)

    return left_warped


#def check_disparity_gt():

if __name__ == "__main__":
    left_path = r"C:\Users\cyzho\Desktop\to_check_disparity\left\2013_05_28_drive_0002_sync_0000004613_left.png"
    # left = Image.open(left_path)
    left_path = r"D:\Masterarbeit\dataset\sceneflow\driving\left\000001_left.png"
    left = cv2.imread(left_path)
    # left = np.array(left)

    right_path = r"C:\Users\cyzho\Desktop\to_check_disparity\right\2013_05_28_drive_0002_sync_0000004613_right.png"
    # right = Image.open(right_path
    right_path = r"D:\Masterarbeit\dataset\sceneflow\driving\right\000001_right.png"
    right = cv2.imread(right_path)
    # right = np.array(right)

    disp_path = r"C:\Users\cyzho\Desktop\to_check_disparity\disparity\2013_05_28_drive_0002_sync_0000004613_disparity.tiff"
    disp_path = r"D:\Masterarbeit\dataset\sceneflow\driving\disparity\000001_disparity.tiff"
    disp = Image.open(disp_path)
    disp = np.array(disp)

    direction_str = 'l2r'
    #warped_img = to_warped_image(left, disp, direction_str)
    direction_str = 'r2l'
    warped_img = to_warped_image(right, disp, direction_str)
    cv2.imshow("Image", warped_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
