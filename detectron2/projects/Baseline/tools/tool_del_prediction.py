#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@Project ：Masterarbeit
@File    ：tool_del_prediction.py
@Author  ：Yu Cao
@Date    ：2021/12/22 8:21 
"""
import os

for root, dirs, files in os.walk(r'C:\Users\cyzho\Desktop\demo_test\prediction_kitti2015'):
    for file in files:
        if os.path.splitext(file)[0][-1] == '1':
            os.remove(os.path.join(root, file))
