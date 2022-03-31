#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@Project ：Masterarbeit
@File    ：tool_del_prediction.py
@Author  ：Yu Cao
@Date    ：2021/12/22 8:21 
"""
import os

path = r'C:\Users\cyzho\Desktop\demo_test\prediction_kitti2015'
path = "/projects/Baseline/datasets/kitti_2015/data_scene_flow/training/image_3"
for root, dirs, files in os.walk(path):
    for file in files:
        if os.path.splitext(file)[0][-1] == '1':
            os.remove(os.path.join(root, file))
