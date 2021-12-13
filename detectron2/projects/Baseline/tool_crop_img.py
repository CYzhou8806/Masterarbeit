#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@Project ：Masterarbeit
@File    ：tool_crop_img.py
@Author  ：Yu Cao
@Date    ：2021/12/11 12:42 
"""

from PIL import Image

path_left = "datasets/kitti_2015/data_scene_flow/training/image_2/000020_10.png"
path_right = "datasets/kitti_2015/data_scene_flow/training/image_3/000020_10.png"


img = Image.open(path_left)
width, height = img.size
# 前两个坐标点是左上角坐标
# 后两个坐标点是右下角坐标
# width在前， height在后
box = (600, 20, 1112, 276)
region = img.crop(box)
region.save(path_left.replace('_10', '_10_crop'))

img = Image.open(path_right)
region = img.crop(box)
region.save(path_right.replace('_10', '_10_crop'))