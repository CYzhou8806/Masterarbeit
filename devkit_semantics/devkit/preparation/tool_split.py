#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@Project ：Masterarbeit
@File    ：tool_split.py
@Author  ：Yu Cao
@Date    ：2022/1/2 16:26
"""
import os
import shutil

os.environ['KITTI2015_DATASET'] = "/home/eistrauben/github/Masterarbeit/detectron2/projects/Baseline/datasets/kitti_2015"
kitti2015Path = os.environ['KITTI2015_DATASET']
kitti2015Path = os.path.join(kitti2015Path, "data_scene_flow")

left_root = os.path.join(kitti2015Path, "image_2")


for root, dirs, files in os.walk(left_root):
    for i, file in enumerate(files):
        if (i+1) % 4 == 0:
            left_file = os.path.join(root, file)
            right_file = left_file.replace("image_2", "image_3")
            gt = left_file.replace("image_2", "disp_occ_0")
            instance = left_file.replace("image_2", "instanceIds").split('.')[0]+"_instanceIds.png"

            shutil.move(left_file, left_file.replace("training", "test"))
            shutil.move(right_file, right_file.replace("training", "test"))
            shutil.move(gt, gt.replace("training", "test"))
            shutil.move(instance, instance.replace("training", "test"))



