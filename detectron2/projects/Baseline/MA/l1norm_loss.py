#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@Project ：Masterarbeit
@File    ：l1norm_loss.py
@Author  ：Yu Cao
@Date    ：2021/12/15 16:44 
"""
import torch


def l1_norm_loss(prediction, gt, mask):
    diff = torch.abs(prediction - gt)
    diff_nz = diff[mask]

    if len(diff_nz) == 0:
        return torch.tensor(0.0)
    else:
        return torch.mean(diff_nz)

