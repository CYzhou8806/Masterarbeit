#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@Project ：Masterarbeit
@File    ：crop_dataset.py
@Author  ：Yu Cao
@Date    ：2022/1/16 10:53 
"""
import os
import shutil
from tqdm import tqdm
from PIL import Image

path_left = "/home/eistrauben/github/Masterarbeit/detectron2/projects/Baseline/datasets/kitti_2015_orig/data_scene_flow/training/image_2"
path_right = "/home/eistrauben/github/Masterarbeit/detectron2/projects/Baseline/datasets/kitti_2015_orig/data_scene_flow/training/image_3"
gt = "/home/eistrauben/github/Masterarbeit/detectron2/projects/Baseline/datasets/kitti_2015_orig/data_scene_flow/training/disp_occ_0"

save_root = "/home/eistrauben/github/Masterarbeit/detectron2/projects/Baseline/datasets/kitti_2015"


# 前两个坐标点是左上角坐标
# 后两个坐标点是右下角坐标
# width在前， height在后
box = (12, 54, 1228, 374)
for root, dirs, files in os.walk(path_left):
    for file in tqdm(files):
        if os.path.splitext(file)[-1] == ".png":
            left_img = os.path.join(root, file)
            right_img = left_img.replace("image_2", "image_3")
            gt = left_img.replace("image_2", "disp_occ_0")

            for to_crop in [left_img, right_img, gt]:
                img = Image.open(to_crop)
                width, height = img.size
                region = img.crop(box)
                region.save(to_crop.replace("kitti_2015_orig", "kitti_2015"))
