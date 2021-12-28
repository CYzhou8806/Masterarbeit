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
from tqdm import tqdm


def to_warped_image(img, disp, direction_str, right):
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

    old_map_x = map_x
    map_x = map_x + disp * direction
    mask = old_map_x == map_x
    assert not np.all(mask)

    for j in range(map_y.shape[1]):
        map_y[:, j] = [y for y in range(map_y.shape[0])]

    left_warped = cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR)

    left_new = np.array(left_warped)
    left_new[mask] = 0
    cv2.imwrite("1.jpg", left_new)
    left_points = cv2.imread("1.jpg")

    right_array = np.array(right)
    right_array[mask] = 0
    cv2.imwrite("2.jpg", right_array)
    right_points = cv2.imread("2.jpg")

    print(np.sum(left_new) - np.sum(mask)*3)

    return left_warped, left_points, right_points


# def check_disparity_gt():

if __name__ == "__main__":
    left_path = r"C:\Users\cyzho\Desktop\to_check_disparity\left\2013_05_28_drive_0002_sync_0000004613_left.png"
    # left = Image.open(left_path)
    left_path = r"D:\Masterarbeit\dataset\sceneflow\driving\left\000001_left.png"
    left_path = r"D:\Masterarbeit\dataset\kitti 2015\kitti_2015\data_scene_flow\training\image_2\000000_10.png"
    left = cv2.imread(left_path)
    # left = np.array(left)

    right_path = r"C:\Users\cyzho\Desktop\to_check_disparity\right\2013_05_28_drive_0002_sync_0000004613_right.png"
    # right = Image.open(right_path
    right_path = r"D:\Masterarbeit\dataset\sceneflow\driving\right\000001_right.png"
    right_path = r"D:\Masterarbeit\dataset\kitti 2015\kitti_2015\data_scene_flow\training\image_3\000000_10.png"
    right = cv2.imread(right_path)
    # right = np.array(right)

    disp_path = r"C:\Users\cyzho\Desktop\to_check_disparity\disparity\2013_05_28_drive_0002_sync_0000004613_disparity.tiff"
    disp_path = r"D:\Masterarbeit\dataset\sceneflow\driving\disparity\000001_disparity.tiff"
    disp_path = r"D:\Masterarbeit\dataset\kitti 2015\kitti_2015\data_scene_flow\training\disp_occ_0\000000_10.png"
    #disp_path = r"C:\Users\cyzho\Desktop\result_dis.png"

    disp = Image.open(disp_path)
    disp = np.array(disp, dtype=np.float32)
    disp = disp/256.0

    direction_str = 'l2r'
    warped_img, warped_points, right_points = to_warped_image(left, disp, direction_str, right)
    direction_str = 'r2l'
    # warped_img, warped_points, right_points = to_warped_image(right, disp, direction_str,left)

    left_warped_points = np.array(warped_points)
    right_points = np.array(right_points)


    diff_positions = []
    for c in [0, 1, 2]:
        for x in tqdm(range(left_warped_points.shape[0])):
            for y in tqdm(range(left_warped_points.shape[1])):
                if left_warped_points[x, y, c] != right_points[x, y, c]:
                    '''
                    print(left_warped_points[x, y, c])
                    print('\n')
                    print(right_points[x, y, c])
                    print('\n')
                    print('\n')
                    print('\n')
                    '''
                    if abs(left_warped_points[x, y, c] - right_points[x, y, c]) < 1:
                        diff_positions.append([x, y])
                else:
                    diff_positions.append([x, y])

    print(np.all(left_warped_points == right_points))
    print(np.sum(left_warped_points != right_points))
    print(np.max(abs(left_warped_points-right_points)))

    for [x,y] in tqdm(diff_positions):
        left_warped_points[x,y] = 0

    cv2.imwrite("3.jpg", left_warped_points)
    left_differ = cv2.imread("3.jpg")



    print('stop')

    cv2.imshow("Image", left_differ)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

