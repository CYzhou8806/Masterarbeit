#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@Project ：Masterarbeit
@File    ：__init__.py.py
@Author  ：Yu Cao
@Date    ：2021/11/21 8:43 
"""

from .config import add_joint_estimation_config
from .panoptic_seg import (
    PanopticDeepLab,
    INS_EMBED_BRANCHES_REGISTRY,
    build_ins_embed_branch,
    PanopticDeepLabSemSegHead,
    PanopticDeepLabInsEmbedHead,
)
from .network import (
    JointEstimation,
    JointEstimationSemSegHead,
    build_dis_embed_head,
)

from .cityscapes_jointestamation import register_all_cityscapes_joint
from .joint_dataset_mapper import JointDeeplabDatasetMapper
from .sceneflow_driving_dispar import register_all_sceneflow
from .kitti_2015_dispar import register_all_kitti_2015
# from l1norm_loss import l1_norm_loss
