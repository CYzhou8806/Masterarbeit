#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@Project ：Masterarbeit
@File    ：tools.py
@Author  ：Yu Cao
@Date    ：2021/12/1 19:19 
"""
import os

import cv2


def down_samples_dataset(dataset_root, output_root, scale):
    for root, dirs, files in os.walk(dataset_root):
        for file in files:
            if os.path.splitext(file)[-1] == ".png":
                pass


img =
pic = cv2.imread(img)

'''
pic_n = cv2.resize(pic, (1280, 720))
pic_name = i
cv2.imwrite(os.path.join(resultDir, i), pic_n)
'''
