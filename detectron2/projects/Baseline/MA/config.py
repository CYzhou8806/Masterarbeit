# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.

from detectron2.config import CfgNode as CN
from detectron2.projects.deeplab import add_deeplab_config


def add_joint_estimation_config(cfg):
    """
    Add config for Panoptic-DeepLab.
    """
    # Reuse DeepLab config.
    add_deeplab_config(cfg)
    # Target generation parameters.
    cfg.INPUT.GAUSSIAN_SIGMA = 10
    cfg.INPUT.IGNORE_STUFF_IN_OFFSET = True
    cfg.INPUT.SMALL_INSTANCE_AREA = 4096
    cfg.INPUT.SMALL_INSTANCE_WEIGHT = 3
    cfg.INPUT.IGNORE_CROWD_IN_SEMANTIC = False
    cfg.INPUT.IMG_SIZE = [1024, 2048]
    cfg.INPUT.DO_AUGUMENTATION = False
    # Optimizer type.
    cfg.SOLVER.OPTIMIZER = "ADAM"
    # Panoptic-DeepLab semantic segmentation head.
    # We add an extra convolution before predictor.

    cfg.MODEL.SEM_SEG_HEAD.HEAD_CHANNELS = 256
    cfg.MODEL.SEM_SEG_HEAD.LOSS_TOP_K = 0.2
    # Panoptic-DeepLab instance segmentation head.
    cfg.MODEL.INS_EMBED_HEAD = CN()
    cfg.MODEL.INS_EMBED_HEAD.NAME = "PanopticDeepLabInsEmbedHead"
    cfg.MODEL.INS_EMBED_HEAD.IN_FEATURES = ["res2", "res3", "res5"]
    cfg.MODEL.INS_EMBED_HEAD.PROJECT_FEATURES = ["res2", "res3"]
    cfg.MODEL.INS_EMBED_HEAD.PROJECT_CHANNELS = [32, 64]
    cfg.MODEL.INS_EMBED_HEAD.ASPP_CHANNELS = 256
    cfg.MODEL.INS_EMBED_HEAD.ASPP_DILATIONS = [6, 12, 18]
    cfg.MODEL.INS_EMBED_HEAD.ASPP_DROPOUT = 0.1
    # We add an extra convolution before predictor.
    cfg.MODEL.INS_EMBED_HEAD.HEAD_CHANNELS = 32
    cfg.MODEL.INS_EMBED_HEAD.CONVS_DIM = 128
    cfg.MODEL.INS_EMBED_HEAD.COMMON_STRIDE = 4
    cfg.MODEL.INS_EMBED_HEAD.NORM = "SyncBN"
    cfg.MODEL.INS_EMBED_HEAD.CENTER_LOSS_WEIGHT = 200.0
    cfg.MODEL.INS_EMBED_HEAD.OFFSET_LOSS_WEIGHT = 0.01
    cfg.MODEL.INS_EMBED_HEAD.NUM_CLASSES = None

    # JointEstimation Disparity Head.
    cfg.MODEL.DIS_EMBED_HEAD = CN()
    cfg.MODEL.DIS_EMBED_HEAD.NAME = "JointEstimationDisEmbedHead"
    cfg.MODEL.DIS_EMBED_HEAD.IN_FEATURES = ["res2", "res3", "res5"]
    cfg.MODEL.DIS_EMBED_HEAD.PROJECT_FEATURES = ["res2", "res3"]
    cfg.MODEL.DIS_EMBED_HEAD.PROJECT_CHANNELS = [32, 64]
    cfg.MODEL.DIS_EMBED_HEAD.ASPP_CHANNELS = 256
    cfg.MODEL.DIS_EMBED_HEAD.ASPP_DILATIONS = [6, 12, 18]
    cfg.MODEL.DIS_EMBED_HEAD.ASPP_DROPOUT = 0.1
    # We add an extra convolution before predictor.
    cfg.MODEL.DIS_EMBED_HEAD.HEAD_CHANNELS = 256
    cfg.MODEL.DIS_EMBED_HEAD.CONVS_DIM = 128
    cfg.MODEL.DIS_EMBED_HEAD.COMMON_STRIDE = 4
    cfg.MODEL.DIS_EMBED_HEAD.NORM = "SyncBN"
    cfg.MODEL.DIS_EMBED_HEAD.NUM_CLASSES = None
    cfg.MODEL.DIS_EMBED_HEAD.LOSS_WEIGHT = 1.0
    cfg.MODEL.DIS_EMBED_HEAD.IGNORE_VALUE = 255
    cfg.MODEL.DIS_EMBED_HEAD.MAX_DISP = 192
    cfg.MODEL.DIS_EMBED_HEAD.LOSS_TYPE = "panoptic_guided"
    cfg.MODEL.DIS_EMBED_HEAD.HOURGLASS_LOSS_WEIGHT = [0.5, 0.7, 1.0]
    cfg.MODEL.DIS_EMBED_HEAD.INTERNAL_LOSS_WEIGHT = [0.5, 0.7, 1.0]
    cfg.MODEL.DIS_EMBED_HEAD.GUIDED_LOSS_WEIGHT = [0.5, 0.7, 1.0]   # guided_smooth, guided_boundary, smooth_L1
    cfg.MODEL.DIS_EMBED_HEAD.STRESHOLD_GUIDED_LOSS = 1.0
    cfg.MODEL.DIS_EMBED_HEAD.REGRESSION_INPLANES = 256
    cfg.MODEL.DIS_EMBED_HEAD.HOURGLASS_INPLANES = 32
    cfg.MODEL.DIS_EMBED_HEAD.HOURGLASS_TYPE = "hourglass_2D"
    cfg.MODEL.DIS_EMBED_HEAD.RESOL_DISP_ADAPT = False
    cfg.MODEL.DIS_EMBED_HEAD.GRADIENT_TYPE = "sobel"
    cfg.MODEL.DIS_EMBED_HEAD.ZERO_DIS_CONSIDERED = True

    # Panoptic-DeepLab post-processing setting.
    cfg.MODEL.PANOPTIC_DEEPLAB = CN()
    # Stuff area limit, ignore stuff region below this number.
    cfg.MODEL.PANOPTIC_DEEPLAB.STUFF_AREA = 2048
    cfg.MODEL.PANOPTIC_DEEPLAB.CENTER_THRESHOLD = 0.1
    cfg.MODEL.PANOPTIC_DEEPLAB.NMS_KERNEL = 7
    cfg.MODEL.PANOPTIC_DEEPLAB.TOP_K_INSTANCE = 200
    # If set to False, Panoptic-DeepLab will not evaluate instance segmentation.
    cfg.MODEL.PANOPTIC_DEEPLAB.PREDICT_INSTANCES = True
    cfg.MODEL.PANOPTIC_DEEPLAB.USE_DEPTHWISE_SEPARABLE_CONV = False
    # This is the padding parameter for images with various sizes. ASPP layers
    # requires input images to be divisible by the average pooling size and we
    # can use `MODEL.PANOPTIC_DEEPLAB.SIZE_DIVISIBILITY` to pad all images to
    # a fixed resolution (e.g. 640x640 for COCO) to avoid having a image size
    # that is not divisible by ASPP average pooling size.
    cfg.MODEL.PANOPTIC_DEEPLAB.SIZE_DIVISIBILITY = -1
    # Only evaluates network speed (ignores post-processing).
    cfg.MODEL.PANOPTIC_DEEPLAB.BENCHMARK_NETWORK_SPEED = False
    cfg.MODEL.MODE = CN()
    cfg.MODEL.MODE.PANOPTIC_BRANCH = True
    cfg.MODEL.MODE.DISPARITY_BRANCH = True
    cfg.MODEL.MODE.FEATURE_FUSION = True
    cfg.RESUME = False
    cfg.SOLVER.FREEZE_BACKBONE = False
    cfg.SOLVER.FREEZE_PANOPTIC = False
    cfg.SOLVER.FREEZE_DISPARITY = False
    cfg.SOLVER.FREEZE_DISPARITY_8 = False
    cfg.SOLVER.FREEZE_DISPARITY_16 = False
    cfg.SOLVER.FREEZE_DISPARITY_4 = False
    cfg.INPUT.CROP.VAL_SIZE = (320,1216)
    cfg.MODEL.DIS_EMBED_HEAD.FUSION_MODEL = "multi"
    cfg.MODEL.DIS_EMBED_HEAD.ADAPTIVE_MOD = "all_adaptive"

