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

path_left = "/home/eistrauben/github/Masterarbeit/detectron2/projects/Baseline/datasets/kitti_2015/data_scene_flow/val/image_2"

# 前两个坐标点是左上角坐标
# 后两个坐标点是右下角坐标
# width在前， height在后
box1 = (0, 119, 512, 375)
box2 = (365, 119, 877, 375)
box3 = (730, 119, 1242, 375)

for root, dirs, files in os.walk(path_left):
    for file in tqdm(files):
        if os.path.splitext(file)[-1] == ".png":
            left_img = os.path.join(root, file)
            right_img = left_img.replace("image_2", "image_3")
            gt = left_img.replace("image_2", "disp_occ_0")
            gt_instance = left_img.replace("image_2", "instanceIds").split('.')[0] + "_instanceIds.png"

            for to_crop in [left_img, right_img, gt, gt_instance]:
                img = Image.open(to_crop)
                width, height = img.size
                basename = os.path.basename(to_crop)
                for n, box in zip(['1', '2', '3'], [box1, box2, box3]):
                    region = img.crop(box)
                    new_name = n + basename[1:]
                    save_path = os.path.join(os.path.dirname(to_crop.replace("val", "val_crop")), new_name)
                    if not os.path.exists(os.path.dirname(save_path)):
                        os.makedirs(os.path.dirname(save_path))
                    region.save(save_path)
