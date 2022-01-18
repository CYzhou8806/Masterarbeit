#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@Project ：Masterarbeit
@File    ：tool_crop_img.py
@Author  ：Yu Cao
@Date    ：2021/12/11 12:42 
"""

from PIL import Image

path_left = "/home/eistrauben/github/Masterarbeit/detectron2/projects/Baseline/datasets/kitti_2015/data_scene_flow/training/image_2/000125_10.png"
path_right = "/home/eistrauben/github/Masterarbeit/detectron2/projects/Baseline/datasets/kitti_2015/data_scene_flow/training/image_3/000125_10.png"
gt = "/home/eistrauben/github/Masterarbeit/detectron2/projects/Baseline/datasets/kitti_2015/data_scene_flow/training/disp_occ_0/000125_10.png"

img = Image.open(path_left)
width, height = img.size
# 前两个坐标点是左上角坐标
# 后两个坐标点是右下角坐标
# width在前， height在后
box = (1, 127, 513, 383)
region = img.crop(box)
region.save(path_left.replace('_10', '_10_crop1'))

img = Image.open(path_right)
region = img.crop(box)
region.save(path_right.replace('_10', '_10_crop1'))

img = Image.open(gt)
region = img.crop(box)
region.save(gt.replace('_10', '_10_crop1'))





img = Image.open(path_left)
# 前两个坐标点是左上角坐标
# 后两个坐标点是右下角坐标
# width在前， height在后
box = (514, 127, 1026, 383)
region = img.crop(box)
region.save(path_left.replace('_10', '_10_crop2'))

img = Image.open(path_right)
region = img.crop(box)
region.save(path_right.replace('_10', '_10_crop2'))

img = Image.open(gt)
region = img.crop(box)
region.save(gt.replace('_10', '_10_crop2'))
