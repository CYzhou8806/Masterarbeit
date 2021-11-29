#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@Project ：Masterarbeit
@File    ：network.py
@Author  ：Yu Cao
@Date    ：2021/11/21 10:30 
"""
import copy
import cv2 as cv

import torch.utils.data
import math
import numpy as np
from typing import Callable, Dict, List, Union
import fvcore.nn.weight_init as weight_init
import torch
from torch import nn
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.data import MetadataCatalog
from detectron2.layers import Conv2d, DepthwiseSeparableConv2d, ShapeSpec, get_norm
from detectron2.modeling import (
    META_ARCH_REGISTRY,
    SEM_SEG_HEADS_REGISTRY,
    build_backbone,
    build_sem_seg_head,
)
from detectron2.modeling.postprocessing import sem_seg_postprocess
from detectron2.projects.deeplab import DeepLabV3PlusHead
from detectron2.projects.deeplab.loss import DeepLabCE
from detectron2.structures import BitMasks, ImageList, Instances
from detectron2.utils.registry import Registry

from .post_processing import get_panoptic_segmentation
from .submodule import convbn_3d, disparityregression, convbn

__all__ = ["JointEstimation", "INS_EMBED_BRANCHES_REGISTRY", "build_ins_embed_branch", "build_dis_embed_head"]

INS_EMBED_BRANCHES_REGISTRY = Registry("INS_EMBED_BRANCHES")
INS_EMBED_BRANCHES_REGISTRY.__doc__ = """
Registry for instance embedding branches, which make instance embedding
predictions from feature maps.
"""

DIS_EMBED_BRANCHES_REGISTRY = Registry("DIS_EMBED_BRANCHES")
DIS_EMBED_BRANCHES_REGISTRY.__doc__ = """
Registry for disparity embedding branches, which make disparity embedding
predictions from feature maps.
"""


@META_ARCH_REGISTRY.register()
class JointEstimation(nn.Module):
    """
    Main class for joint estimation architectures.
    """

    def __init__(self, cfg):
        super().__init__()
        self.backbone = build_backbone(cfg)  # the shared encoder (without ASPP)

        self.sem_seg_head = build_sem_seg_head(cfg, self.backbone.output_shape())
        self.ins_embed_head = build_ins_embed_branch(cfg, self.backbone.output_shape())
        self.dis_embed_head = build_dis_embed_head(cfg, self.backbone.output_shape())

        self.max_disp = cfg.MODEL.INS_EMBED_HEAD.MAX_DISP

        # TODO: following meaning still not clear
        self.register_buffer("pixel_mean", torch.tensor(cfg.MODEL.PIXEL_MEAN).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.tensor(cfg.MODEL.PIXEL_STD).view(-1, 1, 1), False)
        self.meta = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])
        self.stuff_area = cfg.MODEL.PANOPTIC_DEEPLAB.STUFF_AREA
        self.threshold = cfg.MODEL.PANOPTIC_DEEPLAB.CENTER_THRESHOLD
        self.nms_kernel = cfg.MODEL.PANOPTIC_DEEPLAB.NMS_KERNEL
        self.top_k = cfg.MODEL.PANOPTIC_DEEPLAB.TOP_K_INSTANCE
        self.predict_instances = cfg.MODEL.PANOPTIC_DEEPLAB.PREDICT_INSTANCES
        self.use_depthwise_separable_conv = cfg.MODEL.PANOPTIC_DEEPLAB.USE_DEPTHWISE_SEPARABLE_CONV
        assert (
                cfg.MODEL.SEM_SEG_HEAD.USE_DEPTHWISE_SEPARABLE_CONV
                == cfg.MODEL.PANOPTIC_DEEPLAB.USE_DEPTHWISE_SEPARABLE_CONV
        )
        self.size_divisibility = cfg.MODEL.PANOPTIC_DEEPLAB.SIZE_DIVISIBILITY
        self.benchmark_network_speed = cfg.MODEL.PANOPTIC_DEEPLAB.BENCHMARK_NETWORK_SPEED

    @property
    def device(self):
        return self.pixel_mean.device

    def forward(self, batched_inputs):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper`.
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:
                   * "image": Tensor, image in (C, H, W) format.
                   * "sem_seg": semantic segmentation ground truth
                   * "center": center points heatmap ground truth
                   * "offset": pixel offsets to center points ground truth
                   * Other information that's included in the original dicts, such as:
                     "height", "width" (int): the output resolution of the model (may be different
                     from input resolution), used in inference.
        Returns:
            list[dict]:
                each dict is the results for one image. The dict contains the following keys:

                * "panoptic_seg", "sem_seg": see documentation
                    :doc:`/tutorials/models` for the standard output format
                * "instances": available if ``predict_instances is True``. see documentation
                    :doc:`/tutorials/models` for the standard output format
        """

        # To avoid error in ASPP layer when input has different size.
        size_divisibility = (
            self.size_divisibility
            if self.size_divisibility > 0
            else self.backbone.size_divisibility
        )

        # load left images
        left_images = [x["image"].to(self.device) for x in batched_inputs]
        left_images = [(x - self.pixel_mean) / self.pixel_std for x in left_images]
        left_images = ImageList.from_tensors(left_images, size_divisibility)
        left_features = self.backbone(left_images.tensor)

        # load right images
        right_images = [x["image"].to(self.device) for x in batched_inputs]
        right_images = [(x - self.pixel_mean) / self.pixel_std for x in right_images]
        right_images = ImageList.from_tensors(right_images, size_divisibility)
        right_features = self.backbone(right_images.tensor)

        losses = {}

        # Needs to be recovered
        '''
        # semantic branch
        if "sem_seg" in batched_inputs[0]:
            targets = [x["sem_seg"].to(self.device) for x in batched_inputs]
            targets = ImageList.from_tensors(
                targets, size_divisibility, self.sem_seg_head.ignore_value
            ).tensor
            if "sem_seg_weights" in batched_inputs[0]:
                # The default D2 DatasetMapper may not contain "sem_seg_weights"
                # Avoid error in testing when default DatasetMapper is used.
                weights = [x["sem_seg_weights"].to(self.device) for x in batched_inputs]
                weights = ImageList.from_tensors(weights, size_divisibility).tensor
            else:
                weights = None
        else:
            targets = None
            weights = None
        sem_seg_results, sem_seg_losses, left_sem_seg_features = self.sem_seg_head(left_features, targets, weights)
        losses.update(sem_seg_losses)
        right_sem_seg_results, _, right_sem_seg_features = self.sem_seg_head(right_features, None, None, is_left=False)

        # instance branch
        if "center" in batched_inputs[0] and "offset" in batched_inputs[0]:
            center_targets = [x["center"].to(self.device) for x in batched_inputs]
            center_targets = ImageList.from_tensors(
                center_targets, size_divisibility
            ).tensor.unsqueeze(1)
            center_weights = [x["center_weights"].to(self.device) for x in batched_inputs]
            center_weights = ImageList.from_tensors(center_weights, size_divisibility).tensor
            offset_targets = [x["offset"].to(self.device) for x in batched_inputs]
            offset_targets = ImageList.from_tensors(offset_targets, size_divisibility).tensor
            offset_weights = [x["offset_weights"].to(self.device) for x in batched_inputs]
            offset_weights = ImageList.from_tensors(offset_weights, size_divisibility).tensor
        else:
            center_targets = None
            center_weights = None
            offset_targets = None
            offset_weights = None
        center_results, offset_results, center_losses, offset_losses, left_ins_seg_features = self.ins_embed_head(
            left_features, center_targets, center_weights, offset_targets, offset_weights
        )
        losses.update(center_losses)
        losses.update(offset_losses)
        right_center_results, right_offset_results, _, _, right_ins_seg_features = self.ins_embed_head(
            right_features, None, None, None, None, is_left=False)

        # TODO: convert 256 -> 64
        # dict{'1/4': [[left_seg, right_seg], [left_ins, right_ins], [left_dis, right_dis]], ...}
        pyramid_features = {}
        for key in left_sem_seg_features:
            pyramid_features[key] = []
            pyramid_features[key].append([left_sem_seg_features[key], right_sem_seg_features[key]])
            pyramid_features[key].append([left_ins_seg_features[key], right_ins_seg_features[key]])
            # pyramid_features[key].append([left_dis_features[key], right_dis_features[key]])
        self.dis_embed_head(left_features, right_features, pyramid_features)
        '''

        pyramid_features = {}
        self.dis_embed_head(left_features, right_features, pyramid_features)

        '''
        # tmp
        print(left_sem_seg_features['1/16'].size())
        seg_cost_volume = build_correlation_cost_volume(192, left_sem_seg_features['1/16'], right_sem_seg_features['1/16'])
        print(seg_cost_volume.size())
        cost_volume = seg_cost_volume * seg_cost_volume
        print(cost_volume.size())
        raise RuntimeError('excepted stop')
        '''

        if self.training:
            return losses

        if self.benchmark_network_speed:
            return []

        # to be recovered
        '''
        processed_results = []
        for sem_seg_result, center_result, offset_result, input_per_image, image_size in zip(
                sem_seg_results, center_results, offset_results, batched_inputs, left_images.image_sizes
        ):
            height = input_per_image.get("height")
            width = input_per_image.get("width")
            r = sem_seg_postprocess(sem_seg_result, image_size, height, width)
            c = sem_seg_postprocess(center_result, image_size, height, width)
            o = sem_seg_postprocess(offset_result, image_size, height, width)
            # Post-processing to get panoptic segmentation.
            panoptic_image, _ = get_panoptic_segmentation(
                r.argmax(dim=0, keepdim=True),
                c,
                o,
                thing_ids=self.meta.thing_dataset_id_to_contiguous_id.values(),
                label_divisor=self.meta.label_divisor,
                stuff_area=self.stuff_area,
                void_label=-1,
                threshold=self.threshold,
                nms_kernel=self.nms_kernel,
                top_k=self.top_k,
            )
            # For semantic segmentation evaluation.
            processed_results.append({"sem_seg": r})
            panoptic_image = panoptic_image.squeeze(0)
            semantic_prob = F.softmax(r, dim=0)
            # For panoptic segmentation evaluation.
            processed_results[-1]["panoptic_seg"] = (panoptic_image, None)
            # For instance segmentation evaluation.
            if self.predict_instances:
                instances = []
                panoptic_image_cpu = panoptic_image.cpu().numpy()
                for panoptic_label in np.unique(panoptic_image_cpu):
                    if panoptic_label == -1:
                        continue
                    pred_class = panoptic_label // self.meta.label_divisor
                    isthing = pred_class in list(
                        self.meta.thing_dataset_id_to_contiguous_id.values()
                    )
                    # Get instance segmentation results.
                    if isthing:
                        instance = Instances((height, width))
                        # Evaluation code takes continuous id starting from 0
                        instance.pred_classes = torch.tensor(
                            [pred_class], device=panoptic_image.device
                        )
                        mask = panoptic_image == panoptic_label
                        instance.pred_masks = mask.unsqueeze(0)
                        # Average semantic probability
                        sem_scores = semantic_prob[pred_class, ...]
                        sem_scores = torch.mean(sem_scores[mask])
                        # Center point probability
                        mask_indices = torch.nonzero(mask).float()
                        center_y, center_x = (
                            torch.mean(mask_indices[:, 0]),
                            torch.mean(mask_indices[:, 1]),
                        )
                        center_scores = c[0, int(center_y.item()), int(center_x.item())]
                        # Confidence score is semantic prob * center prob.
                        instance.scores = torch.tensor(
                            [sem_scores * center_scores], device=panoptic_image.device
                        )
                        # Get bounding boxes
                        instance.pred_boxes = BitMasks(instance.pred_masks).get_bounding_boxes()
                        instances.append(instance)
                if len(instances) > 0:
                    processed_results[-1]["instances"] = Instances.cat(instances)

        return processed_results
        '''


@SEM_SEG_HEADS_REGISTRY.register()
class JointEstimationSemSegHead(DeepLabV3PlusHead):
    """
    A semantic segmentation head of joint estimation architectures`.
    """

    @configurable
    def __init__(
            self,
            input_shape: Dict[str, ShapeSpec],
            *,
            decoder_channels: List[int],
            norm: Union[str, Callable],
            head_channels: int,
            loss_weight: float,
            loss_type: str,
            loss_top_k: float,
            ignore_value: int,
            num_classes: int,
            **kwargs,
    ):
        """
        NOTE: this interface is experimental.

        Args:
            input_shape (ShapeSpec): shape of the input feature
            decoder_channels (list[int]): a list of output channels of each
                decoder stage. It should have the same length as "input_shape"
                (each element in "input_shape" corresponds to one decoder stage).
            norm (str or callable): normalization for all conv layers.
            head_channels (int): the output channels of extra convolutions
                between decoder and predictor.
            loss_weight (float): loss weight.
            loss_top_k: (float): setting the top k% hardest pixels for
                "hard_pixel_mining" loss.
            loss_type, ignore_value, num_classes: the same as the base class.
        """
        super().__init__(
            input_shape,
            decoder_channels=decoder_channels,
            norm=norm,
            ignore_value=ignore_value,
            **kwargs,
        )
        assert self.decoder_only

        self.loss_weight = loss_weight
        use_bias = norm == ""
        # `head` is additional transform before predictor
        if self.use_depthwise_separable_conv:
            # We use a single 5x5 DepthwiseSeparableConv2d to replace
            # 2 3x3 Conv2d since they have the same receptive field.
            self.head = DepthwiseSeparableConv2d(
                decoder_channels[0],
                head_channels,
                kernel_size=5,
                padding=2,
                norm1=norm,
                activation1=F.relu,
                norm2=norm,
                activation2=F.relu,
            )
        else:
            self.head = nn.Sequential(
                Conv2d(
                    decoder_channels[0],
                    decoder_channels[0],
                    kernel_size=3,
                    padding=1,
                    bias=use_bias,
                    norm=get_norm(norm, decoder_channels[0]),
                    activation=F.relu,
                ),
                Conv2d(
                    decoder_channels[0],
                    head_channels,
                    kernel_size=3,
                    padding=1,
                    bias=use_bias,
                    norm=get_norm(norm, head_channels),
                    activation=F.relu,
                ),
            )
            weight_init.c2_xavier_fill(self.head[0])
            weight_init.c2_xavier_fill(self.head[1])
        self.predictor = Conv2d(head_channels, num_classes, kernel_size=1)
        nn.init.normal_(self.predictor.weight, 0, 0.001)
        nn.init.constant_(self.predictor.bias, 0)

        if loss_type == "cross_entropy":
            self.loss = nn.CrossEntropyLoss(reduction="mean", ignore_index=ignore_value)
        elif loss_type == "hard_pixel_mining":
            self.loss = DeepLabCE(ignore_label=ignore_value, top_k_percent_pixels=loss_top_k)
        else:
            raise ValueError("Unexpected loss type: %s" % loss_type)

    @classmethod
    def from_config(cls, cfg, input_shape):
        ret = super().from_config(cfg, input_shape)
        ret["head_channels"] = cfg.MODEL.SEM_SEG_HEAD.HEAD_CHANNELS
        ret["loss_top_k"] = cfg.MODEL.SEM_SEG_HEAD.LOSS_TOP_K
        return ret

    def forward(self, features, targets=None, weights=None, is_left=True):
        """
        Returns:
            In training, returns (None, dict of losses)
            In inference, returns (CxHxW logits, {})
        """
        y, out_features = self.layers(features)
        if self.training and is_left:
            return None, self.losses(y, targets, weights), out_features
        else:
            y = F.interpolate(
                y, scale_factor=self.common_stride, mode="bilinear", align_corners=False
            )
            return y, {}, out_features

    def layers(self, features):
        assert self.decoder_only
        out_features = {}
        # Reverse feature maps into top-down order (from low to high resolution)
        for i, f in enumerate(self.in_features[::-1]):
            x = features[f]  # "features" is dictionary
            proj_x = self.decoder[f]["project_conv"](x)
            if self.decoder[f]["fuse_conv"] is None:
                # This is aspp module
                y = proj_x
            else:
                # Upsample y
                y = F.interpolate(y, size=proj_x.size()[2:], mode="bilinear", align_corners=False)
                y = torch.cat([proj_x, y], dim=1)
                y = self.decoder[f]["fuse_conv"](y)

            # save outputs
            if i == 1:
                out_features['1/8'] = y
            elif i == 2:
                out_features['1/4'] = y
            elif i == 0:
                out_features['1/16'] = y
            else:
                raise ValueError("undefined output of SemSeg Branch")

        y = out_features['1/4']
        y = self.head(y)
        y = self.predictor(y)
        return y, out_features

    def losses(self, predictions, targets, weights=None):
        predictions = F.interpolate(
            predictions, scale_factor=self.common_stride, mode="bilinear", align_corners=False
        )
        loss = self.loss(predictions, targets, weights)
        losses = {"loss_sem_seg": loss * self.loss_weight}
        return losses


def build_ins_embed_branch(cfg, input_shape):
    """
    Build a instance embedding branch from `cfg.MODEL.INS_EMBED_HEAD.NAME`.
    """
    name = cfg.MODEL.INS_EMBED_HEAD.NAME
    return INS_EMBED_BRANCHES_REGISTRY.get(name)(cfg, input_shape)


@INS_EMBED_BRANCHES_REGISTRY.register()
class JointEstimationInsEmbedHead(DeepLabV3PlusHead):
    """
    A instance embedding head described in :paper:`Panoptic-DeepLab`.
    """

    @configurable
    def __init__(
            self,
            input_shape: Dict[str, ShapeSpec],
            *,
            decoder_channels: List[int],
            norm: Union[str, Callable],
            head_channels: int,
            center_loss_weight: float,
            offset_loss_weight: float,
            **kwargs,
    ):
        """
        NOTE: this interface is experimental.

        Args:
            input_shape (ShapeSpec): shape of the input feature
            decoder_channels (list[int]): a list of output channels of each
                decoder stage. It should have the same length as "input_shape"
                (each element in "input_shape" corresponds to one decoder stage).
            norm (str or callable): normalization for all conv layers.
            head_channels (int): the output channels of extra convolutions
                between decoder and predictor.
            center_loss_weight (float): loss weight for center point prediction.
            offset_loss_weight (float): loss weight for center offset prediction.
        """
        super().__init__(input_shape, decoder_channels=decoder_channels, norm=norm, **kwargs)
        assert self.decoder_only

        self.center_loss_weight = center_loss_weight
        self.offset_loss_weight = offset_loss_weight
        use_bias = norm == ""
        # center prediction
        # `head` is additional transform before predictor
        self.center_head = nn.Sequential(
            Conv2d(
                decoder_channels[0],
                decoder_channels[0],
                kernel_size=3,
                padding=1,
                bias=use_bias,
                norm=get_norm(norm, decoder_channels[0]),
                activation=F.relu,
            ),
            Conv2d(
                decoder_channels[0],
                head_channels,
                kernel_size=3,
                padding=1,
                bias=use_bias,
                norm=get_norm(norm, head_channels),
                activation=F.relu,
            ),
        )
        weight_init.c2_xavier_fill(self.center_head[0])
        weight_init.c2_xavier_fill(self.center_head[1])
        self.center_predictor = Conv2d(head_channels, 1, kernel_size=1)
        nn.init.normal_(self.center_predictor.weight, 0, 0.001)
        nn.init.constant_(self.center_predictor.bias, 0)

        # offset prediction
        # `head` is additional transform before predictor
        if self.use_depthwise_separable_conv:
            # We use a single 5x5 DepthwiseSeparableConv2d to replace
            # 2 3x3 Conv2d since they have the same receptive field.
            self.offset_head = DepthwiseSeparableConv2d(
                decoder_channels[0],
                head_channels,
                kernel_size=5,
                padding=2,
                norm1=norm,
                activation1=F.relu,
                norm2=norm,
                activation2=F.relu,
            )
        else:
            self.offset_head = nn.Sequential(
                Conv2d(
                    decoder_channels[0],
                    decoder_channels[0],
                    kernel_size=3,
                    padding=1,
                    bias=use_bias,
                    norm=get_norm(norm, decoder_channels[0]),
                    activation=F.relu,
                ),
                Conv2d(
                    decoder_channels[0],
                    head_channels,
                    kernel_size=3,
                    padding=1,
                    bias=use_bias,
                    norm=get_norm(norm, head_channels),
                    activation=F.relu,
                ),
            )
            weight_init.c2_xavier_fill(self.offset_head[0])
            weight_init.c2_xavier_fill(self.offset_head[1])
        self.offset_predictor = Conv2d(head_channels, 2, kernel_size=1)
        nn.init.normal_(self.offset_predictor.weight, 0, 0.001)
        nn.init.constant_(self.offset_predictor.bias, 0)

        self.center_loss = nn.MSELoss(reduction="none")
        self.offset_loss = nn.L1Loss(reduction="none")

    @classmethod
    def from_config(cls, cfg, input_shape):
        if cfg.INPUT.CROP.ENABLED:
            assert cfg.INPUT.CROP.TYPE == "absolute"
            train_size = cfg.INPUT.CROP.SIZE
        else:
            train_size = None
        decoder_channels = [cfg.MODEL.INS_EMBED_HEAD.CONVS_DIM] * (
                len(cfg.MODEL.INS_EMBED_HEAD.IN_FEATURES) - 1
        ) + [cfg.MODEL.INS_EMBED_HEAD.ASPP_CHANNELS]
        ret = dict(
            input_shape={
                k: v for k, v in input_shape.items() if k in cfg.MODEL.INS_EMBED_HEAD.IN_FEATURES
            },
            project_channels=cfg.MODEL.INS_EMBED_HEAD.PROJECT_CHANNELS,
            aspp_dilations=cfg.MODEL.INS_EMBED_HEAD.ASPP_DILATIONS,
            aspp_dropout=cfg.MODEL.INS_EMBED_HEAD.ASPP_DROPOUT,
            decoder_channels=decoder_channels,
            common_stride=cfg.MODEL.INS_EMBED_HEAD.COMMON_STRIDE,
            norm=cfg.MODEL.INS_EMBED_HEAD.NORM,
            train_size=train_size,
            head_channels=cfg.MODEL.INS_EMBED_HEAD.HEAD_CHANNELS,
            center_loss_weight=cfg.MODEL.INS_EMBED_HEAD.CENTER_LOSS_WEIGHT,
            offset_loss_weight=cfg.MODEL.INS_EMBED_HEAD.OFFSET_LOSS_WEIGHT,
            use_depthwise_separable_conv=cfg.MODEL.SEM_SEG_HEAD.USE_DEPTHWISE_SEPARABLE_CONV,
        )
        return ret

    def forward(
            self,
            features,
            center_targets=None,
            center_weights=None,
            offset_targets=None,
            offset_weights=None,
            is_left=True
    ):
        """
        Returns:
            In training, returns (None, dict of losses)
            In inference, returns (CxHxW logits, {})
        """
        center, offset, out_features = self.layers(features)
        if self.training and is_left:
            return (
                None,
                None,
                self.center_losses(center, center_targets, center_weights),
                self.offset_losses(offset, offset_targets, offset_weights),
                out_features,
            )
        else:
            center = F.interpolate(
                center, scale_factor=self.common_stride, mode="bilinear", align_corners=False
            )
            offset = (
                    F.interpolate(
                        offset, scale_factor=self.common_stride, mode="bilinear", align_corners=False
                    )
                    * self.common_stride
            )
            return center, offset, {}, {}, out_features

    def layers(self, features):
        assert self.decoder_only
        out_features = {}
        # Reverse feature maps into top-down order (from low to high resolution)
        for i, f in enumerate(self.in_features[::-1]):
            x = features[f]  # "features" is dictionary
            proj_x = self.decoder[f]["project_conv"](x)
            if self.decoder[f]["fuse_conv"] is None:
                # This is aspp module
                y = proj_x
            else:
                # Upsample y
                y = F.interpolate(y, size=proj_x.size()[2:], mode="bilinear", align_corners=False)
                y = torch.cat([proj_x, y], dim=1)
                y = self.decoder[f]["fuse_conv"](y)

            # save outputs
            if i == 1:
                out_features['1/8'] = y
            elif i == 2:
                out_features['1/4'] = y
            elif i == 0:
                out_features['1/16'] = y
            else:
                raise ValueError("undefined output of SemSeg Branch")
        y = out_features['1/4']
        # center
        center = self.center_head(y)
        center = self.center_predictor(center)
        # offset
        offset = self.offset_head(y)
        offset = self.offset_predictor(offset)
        return center, offset, out_features

    def center_losses(self, predictions, targets, weights):
        predictions = F.interpolate(
            predictions, scale_factor=self.common_stride, mode="bilinear", align_corners=False
        )
        loss = self.center_loss(predictions, targets) * weights
        if weights.sum() > 0:
            loss = loss.sum() / weights.sum()
        else:
            loss = loss.sum() * 0
        losses = {"loss_center": loss * self.center_loss_weight}
        return losses

    def offset_losses(self, predictions, targets, weights):
        predictions = (
                F.interpolate(
                    predictions, scale_factor=self.common_stride, mode="bilinear", align_corners=False
                )
                * self.common_stride
        )
        loss = self.offset_loss(predictions, targets) * weights
        if weights.sum() > 0:
            loss = loss.sum() / weights.sum()
        else:
            loss = loss.sum() * 0
        losses = {"loss_offset": loss * self.offset_loss_weight}
        return losses


def build_dis_embed_head(cfg, input_shape):
    """
    Build a disparity embedding branch from `cfg.MODEL.DIS_EMBED_HEAD.NAME`.
    """
    name = cfg.MODEL.DIS_EMBED_HEAD.NAME
    return DIS_EMBED_BRANCHES_REGISTRY.get(name)(cfg, input_shape)


@DIS_EMBED_BRANCHES_REGISTRY.register()
class JointEstimationDisEmbedHead_star(DeepLabV3PlusHead):
    """
    A semantic segmentation head of joint estimation architectures`.
    """

    @configurable
    def __init__(
            self,
            input_shape: Dict[str, ShapeSpec],
            *,
            decoder_channels: List[int],
            norm: Union[str, Callable],
            head_channels: int,
            loss_weight: float,  # the weight for the entire section
            loss_type: str,
            ignore_value: int,
            img_size=None,
            max_disp: int,
            hourglass_loss_weight: List[float],
            internal_loss_weight: List[float],
            guided_loss_weight: List[float],
            streshold_guided_loss: float,
            regression_inplanes: int,
            hourglass_inplanes: int,
            hourglass_type: str,
            resol_disp_adapt: bool,
            **kwargs,
    ):
        """
        NOTE: this interface is experimental.

        Args:
            input_shape (ShapeSpec): shape of the input feature
            decoder_channels (list[int]): a list of output channels of each
                decoder stage. It should have the same length as "input_shape"
                (each element in "input_shape" corresponds to one decoder stage).
            norm (str or callable): normalization for all conv layers.
            head_channels (int): the output channels of extra convolutions
                between decoder and predictor.
            loss_weight (float): loss weight.
            loss_top_k: (float): setting the top k% hardest pixels for
                "hard_pixel_mining" loss.
            loss_type, ignore_value, num_classes: the same as the base class.
        """
        super().__init__(
            input_shape,
            decoder_channels=decoder_channels,
            norm=norm,
            ignore_value=ignore_value,
            **kwargs,
        )

        self.loss_weight = loss_weight
        self.hourglass_loss_weight = hourglass_loss_weight
        self.internal_loss_weight = internal_loss_weight
        self.guided_loss_weight = guided_loss_weight
        self.max_disp = max_disp
        self.lamda = streshold_guided_loss
        self.loss_type = loss_type
        self.hourglass_type = hourglass_type
        self.resol_disp_adapt = resol_disp_adapt
        use_bias = norm == ""
        # `head` is additional transform before predictor
        if self.use_depthwise_separable_conv:
            # We use a single 5x5 DepthwiseSeparableConv2d to replace
            # 2 3x3 Conv2d since they have the same receptive field.
            self.head = DepthwiseSeparableConv2d(
                decoder_channels[0],
                head_channels,
                kernel_size=5,
                padding=2,
                norm1=norm,
                activation1=F.relu,
                norm2=norm,
                activation2=F.relu,
            )
        else:
            self.head = nn.Sequential(
                Conv2d(
                    decoder_channels[0],
                    decoder_channels[0],
                    kernel_size=3,
                    padding=1,
                    bias=use_bias,
                    norm=get_norm(norm, decoder_channels[0]),
                    activation=F.relu,
                ),
                Conv2d(
                    decoder_channels[0],
                    head_channels,
                    kernel_size=3,
                    padding=1,
                    bias=use_bias,
                    norm=get_norm(norm, head_channels),
                    activation=F.relu,
                ),
            )
            weight_init.c2_xavier_fill(self.head[0])
            weight_init.c2_xavier_fill(self.head[1])

        if img_size is None:
            self.img_size = [1024, 2048]  # h, w

        self.dres0 = {}
        self.dres1 = {}
        self.dres2 = {}
        self.dres3 = {}
        self.dres4 = {}
        self.classif1 = {}
        self.classif2 = {}
        self.classif3 = {}

        if self.hourglass_type == "hourglass_2D":
            zoom = [16, 8, 4]
            for i, scale in enumerate(['1/16', '1/8', '1/4']):
                if self.resol_disp_adapt:
                    max_dis = self.max_disp // zoom[i]
                else:
                    max_dis = self.max_disp // 4
                self.dres0[scale] = nn.Sequential(convbn(max_dis, hourglass_inplanes, 3, 1, 1, 1),
                                                  nn.ReLU(inplace=True),
                                                  convbn(hourglass_inplanes, hourglass_inplanes, 3, 1, 1, 1),
                                                  nn.ReLU(inplace=True))
                self.dres1[scale] = nn.Sequential(convbn(hourglass_inplanes, hourglass_inplanes, 3, 1, 1, 1),
                                                  nn.ReLU(inplace=True),
                                                  convbn(hourglass_inplanes, hourglass_inplanes, 3, 1, 1, 1))
                self.dres2[scale] = hourglass_2d(hourglass_inplanes)
                self.dres3[scale] = hourglass_2d(hourglass_inplanes)
                self.dres4[scale] = hourglass_2d(hourglass_inplanes)

                self.classif1[scale] = nn.Sequential(convbn(hourglass_inplanes, hourglass_inplanes, 3, 1, 1, 1),
                                                     nn.ReLU(inplace=True),
                                                     nn.Conv2d(hourglass_inplanes, 1, kernel_size=3, padding=1,
                                                               stride=1,
                                                               bias=False))
                self.classif2[scale] = nn.Sequential(convbn(hourglass_inplanes, hourglass_inplanes, 3, 1, 1, 1),
                                                     nn.ReLU(inplace=True),
                                                     nn.Conv2d(hourglass_inplanes, 1, kernel_size=3, padding=1,
                                                               stride=1,
                                                               bias=False))
                self.classif3[scale] = nn.Sequential(convbn(hourglass_inplanes, hourglass_inplanes, 3, 1, 1, 1),
                                                     nn.ReLU(inplace=True),
                                                     nn.Conv2d(hourglass_inplanes, 1, kernel_size=3, padding=1,
                                                               stride=1,
                                                               bias=False))
        else:
            raise ValueError("Unexpected hourglass type: %s" % self.hourglass_type)

    @classmethod
    def from_config(cls, cfg, input_shape):
        ret = super().from_config(cfg, input_shape)
        ret["head_channels"] = cfg.MODEL.DIS_EMBED_HEAD.HEAD_CHANNELS
        ret["max_disp"] = cfg.MODEL.DIS_EMBED_HEAD.MAX_DISP
        ret["hourglass_loss_weight"] = cfg.MODEL.DIS_EMBED_HEAD.HOURGLASS_LOSS_WEIGHT
        ret["internal_loss_weight"] = cfg.MODEL.DIS_EMBED_HEAD.INTERNAL_LOSS_WEIGHT
        ret["guided_loss_weight"] = cfg.MODEL.DIS_EMBED_HEAD.GUIDED_LOSS_WEIGHT
        ret["streshold_guided_loss"] = cfg.MODEL.DIS_EMBED_HEAD.STRESHOLD_GUIDED_LOSS
        ret["regression_inplanes"] = cfg.MODEL.DIS_EMBED_HEAD.REGRESSION_INPLANES
        ret["hourglass_inplanes"] = cfg.MODEL.DIS_EMBED_HEAD.HOURGLASS_INPLANES
        ret["hourglass_type"] = cfg.MODEL.DIS_EMBED_HEAD.HOURGLASS_TYPE
        ret["resol_disp_adapt"] = cfg.MODEL.DIS_EMBED_HEAD.RESOL_DISP_ADAPT
        return ret

    def forward(self, features, right_features, pyramid_features, dis_targets=None, weights=None, pan_targets=None):
        """
        Returns:
            In training, returns (None, dict of losses)
            In inference, returns (CxHxW logits, {})
        """
        y, out_features = self.layers(features)
        right_y, right_out_features = self.layers(right_features)

        for key in out_features:
            pyramid_features[key].append([out_features[key], right_out_features[key]])

        disparity = []  # form coarse to fine
        zoom = [16, 8, 4]
        for i, scale in enumerate(['1/16', '1/8', '1/4']):
            if self.resol_disp_adapt:
                max_dis = self.max_disp // zoom[i]
            else:
                max_dis = self.max_disp // 4
            if not len(disparity):
                seg_cost_volume = build_correlation_cost_volume(
                    max_dis, pyramid_features[scale][0][0], pyramid_features[scale][0][1])
                ins_cost_volume = build_correlation_cost_volume(
                    max_dis, pyramid_features[scale][1][0], pyramid_features[scale][1][1])
                dis_cost_volume = build_correlation_cost_volume(
                    max_dis, pyramid_features[scale][2][0], pyramid_features[scale][2][1])
            else:  # TODO: add wrap
                dis = disparity[-1][-1]
                seg_cost_volume = build_correlation_cost_volume(
                    max_dis,
                    warping(dis, pyramid_features[scale][0][0]),
                    warping(dis, pyramid_features[scale][0][1]))
                ins_cost_volume = build_correlation_cost_volume(
                    max_dis,
                    warping(dis, pyramid_features[scale][1][0]),
                    warping(dis, pyramid_features[scale][1][1]))
                # print(seg_cost_volume)
                dis_cost_volume = build_correlation_cost_volume(
                    max_dis,
                    warping(dis, pyramid_features[scale][2][0]),
                    warping(dis, pyramid_features[scale][2][1]))
            cost_volume = seg_cost_volume * ins_cost_volume * dis_cost_volume

            cost0 = self.dres0[scale](cost_volume)
            cost0 = self.dres1[scale](cost0) + cost0
            out1, pre1, post1 = self.dres2[scale](cost0, None, None)
            out1 = out1 + cost0
            out2, pre2, post2 = self.dres3[scale](out1, pre1, post1)
            out2 = out2 + cost0
            out3, pre3, post3 = self.dres4[scale](out2, pre1, post2)
            out3 = out3 + cost0
            cost1 = self.classif1[scale](out1)
            cost2 = self.classif2[scale](out2) + cost1
            cost3 = self.classif3[scale](out3) + cost2

            if self.training:
                cost1 = F.upsample(cost1, [max_dis, self.img_size[0], self.img_size[1]], mode='trilinear')
                cost1 = torch.squeeze(cost1, 1)
                pred1 = F.softmax(cost1, dim=1)
                pred1 = disparityregression(max_dis)(pred1)

                cost2 = F.upsample(cost2, [max_dis, self.img_size[0], self.img_size[1]], mode='trilinear')
                cost2 = torch.squeeze(cost2, 1)
                pred2 = F.softmax(cost2, dim=1)
                pred2 = disparityregression(max_dis)(pred2)

            cost3 = F.upsample(cost3, [max_dis, self.img_size[0], self.img_size[1]], mode='trilinear')
            cost3 = torch.squeeze(cost3, 1)
            pred3 = F.softmax(cost3, dim=1)
            # For your information: This formulation 'softmax(c)' learned "similarity"
            # while 'softmax(-c)' learned 'matching cost' as mentioned in the paper.
            # However, 'c' or '-c' do not affect the performance because feature-based cost volume provided flexibility.
            pred3 = disparityregression(max_dis)(pred3)  # TODO: to determine the size

            if self.training:
                if not len(disparity):
                    disparity.append([pred1, pred2, pred3])
                else:
                    disparity.append([pred1 + dis, pred2 + dis, pred3 + dis])
            else:
                if not len(disparity):
                    disparity.append([pred3])
                else:
                    disparity.append([pred3 + dis])

        if self.training:
            return self.losses(disparity, dis_targets, weights, pan_targets), None  # TODO: to be adapted
        else:
            return {}, disparity

    def layers(self, features):
        out_features = {}
        # Reverse feature maps into top-down order (from low to high resolution)
        for i, f in enumerate(self.in_features[::-1]):
            x = features[f]  # "features" is dictionary
            proj_x = self.decoder[f]["project_conv"](x)
            if self.decoder[f]["fuse_conv"] is None:
                # This is aspp module
                y = proj_x
            else:
                # Upsample y
                y = F.interpolate(y, size=proj_x.size()[2:], mode="bilinear", align_corners=False)
                y = torch.cat([proj_x, y], dim=1)
                y = self.decoder[f]["fuse_conv"](y)

            # save outputs
            if i == 1:
                out_features['1/8'] = y
            elif i == 2:
                out_features['1/4'] = y
            elif i == 0:
                out_features['1/16'] = y
            else:
                raise ValueError("undefined output of SemSeg Branch")

        y = out_features['1/4']

        return y, out_features

    def losses(self, predictions, dis_targets, weights=None, pan_targets=None):
        '''
        predictions = F.interpolate(
            predictions, scale_factor=self.common_stride, mode="bilinear", align_corners=False
        )
        '''
        mask = dis_targets < self.max_disp  # TODO: find out the effect of mask
        mask.detach_()
        loss = None
        if self.loss_type == "panoptic_guided":
            smooth_l1 = 0.0
            for i in range(len(predictions)):
                smooth_l1 = smooth_l1 + self.internal_loss_weight[i] * \
                            (self.hourglass_loss_weight[0] *
                             F.smooth_l1_loss(predictions[i][0][mask], dis_targets[mask], size_average=True) +
                             self.hourglass_loss_weight[1] *
                             F.smooth_l1_loss(predictions[i][1][mask], dis_targets[mask], size_average=True) +
                             self.hourglass_loss_weight[2] *
                             F.smooth_l1_loss(predictions[i][2][mask], dis_targets[mask]))

            pan_2rd_gradiant = cv.Laplacian(pan_targets, cv.CV_32F)
            pan_2rd_gradiant = cv.convertScaleAbs(pan_2rd_gradiant)

            bdry_loss = 0.0
            # TODO: implement the boundary loss
            for i in range(len(predictions)):
                dis_2rd_gradiant = cv.Laplacian(predictions[i][-1][mask], cv.CV_32F)
                dis_2rd_gradiant = cv.convertScaleAbs(dis_2rd_gradiant)

                print(dis_2rd_gradiant.size())
                assert dis_2rd_gradiant.size() == pan_2rd_gradiant.size()
                bdry_sum = 0.0
                count = 0
                # all pixel in the map
                for j in range(dis_2rd_gradiant.size()[0]):
                    for k in range(dis_2rd_gradiant.size()[1]):
                        # TODO: add decision
                        # if pan_2rd_gradiant[j, k] not in [road, sidewalk, vegetation, terrain]
                        count = count + 1
                        bdry_sum = bdry_sum + math.exp(-abs(dis_2rd_gradiant[j, k])) * abs(pan_2rd_gradiant[j, k])
                bdry_loss = bdry_loss + self.internal_loss_weight[i] * bdry_sum / count

            sm_loss = 0.0
            # TODO: implement the smooth loss
            for i in range(len(predictions)):
                dis_2rd_gradiant = cv.Laplacian(predictions[i][-1][mask], cv.CV_32F)
                dis_2rd_gradiant = cv.convertScaleAbs(dis_2rd_gradiant)

                sm_sum = 0.0
                count = 0
                # all pixel in the map
                for j in range(dis_2rd_gradiant.size()[0]):
                    for k in range(dis_2rd_gradiant.size()[1]):
                        # TODO: adapt decision
                        if dis_2rd_gradiant[j, k] < self.lamda:
                            count = count + 1
                            sm_sum = sm_sum + math.exp(-abs(pan_2rd_gradiant[j, k])) * abs(dis_2rd_gradiant[j, k])
                sm_loss = sm_loss + self.internal_loss_weight[i] * sm_sum / count

            loss = self.guided_loss_weight[0] * sm_loss + self.guided_loss_weight[1] * bdry_loss + \
                   self.guided_loss_weight[2] * smooth_l1

        elif self.loss_type == "smoothL1_only":
            for i in range(len(predictions)):
                loss = loss + self.internal_loss_weight[i] * \
                       (self.hourglass_loss_weight[0] *
                        F.smooth_l1_loss(predictions[i][0][mask], dis_targets[mask], size_average=True) +
                        self.hourglass_loss_weight[1] *
                        F.smooth_l1_loss(predictions[i][1][mask], dis_targets[mask], size_average=True) +
                        self.hourglass_loss_weight[2] *
                        F.smooth_l1_loss(predictions[i][2][mask], dis_targets[mask]))
        else:
            raise ValueError("Unexpected loss type: %s" % self.loss_type)

        losses = {"loss_dis": loss * self.loss_weight}
        return losses


def build_correlation_cost_volume(max_disp, left_feature, right_feature):
    cost_volume = left_feature.new_zeros(left_feature.size()[0], max_disp,
                                         left_feature.size()[2], left_feature.size()[3])  # (b, max_disp, h, w)
    for i in range(max_disp):
        if i > 0:
            cost_volume[:, i, :, i:] = (left_feature[:, :, :, i:] * right_feature[:, :, :, :-i]).mean(dim=1)
        else:
            cost_volume[:, i, :, :] = (left_feature * right_feature).mean(dim=1)
    return cost_volume


def warping_old(disp, feature):  # TODO: to add operations
    warped = copy.deepcopy(feature)
    return warped


def warping(disp, img, direction_str='r2l'):
    if direction_str == 'r2l':
        direction = - 1
    elif direction_str == 'l2r':
        direction = 1

    map_x = np.zeros((img.shape[0], img.shape[1]), dtype=np.float32)
    map_y = np.zeros((img.shape[0], img.shape[1]), dtype=np.float32)

    for i in range(map_x.shape[0]):
        map_x[i, :] = [x for x in range(map_x.shape[1])]

    if disp.ndim == 3:
        disp = np.squeeze(disp, axis=-1)

    map_x = map_x + disp * direction

    for j in range(map_y.shape[1]):
        map_y[:, j] = [y for y in range(map_y.shape[0])]

    left_warped = cv.remap(img, map_x, map_y, cv.INTER_LINEAR)

    return left_warped


class hourglass(nn.Module):
    def __init__(self, inplanes):
        super(hourglass, self).__init__()

        self.conv1 = nn.Sequential(convbn_3d(inplanes, inplanes * 2, kernel_size=3, stride=2, pad=1),
                                   nn.ReLU(inplace=True))

        self.conv2 = convbn_3d(inplanes * 2, inplanes * 2, kernel_size=3, stride=1, pad=1)

        self.conv3 = nn.Sequential(convbn_3d(inplanes * 2, inplanes * 2, kernel_size=3, stride=2, pad=1),
                                   nn.ReLU(inplace=True))

        self.conv4 = nn.Sequential(convbn_3d(inplanes * 2, inplanes * 2, kernel_size=3, stride=1, pad=1),
                                   nn.ReLU(inplace=True))

        # note: the conv5 and conv6 is without relu
        self.conv5 = nn.Sequential(
            nn.ConvTranspose3d(inplanes * 2, inplanes * 2, kernel_size=3, padding=1, output_padding=1, stride=2,
                               bias=False),
            nn.BatchNorm3d(inplanes * 2))  # +conv2

        self.conv6 = nn.Sequential(
            nn.ConvTranspose3d(inplanes * 2, inplanes, kernel_size=3, padding=1, output_padding=1, stride=2,
                               bias=False),
            nn.BatchNorm3d(inplanes))  # +x

    def forward(self, x, presqu, postsqu):
        out = self.conv1(x)  # in:1/4 out:1/8
        pre = self.conv2(out)  # in:1/8 out:1/8
        if postsqu is not None:
            pre = F.relu(pre + postsqu, inplace=True)  # the red connection in the figure of paper
        else:
            pre = F.relu(pre, inplace=True)

        out = self.conv3(pre)  # in:1/8 out:1/16
        out = self.conv4(out)  # in:1/16 out:1/16

        if presqu is not None:
            # the green connection
            # if this is not the first hourglass, take the output of pre-conv5 to make the fusion
            # a little different from what is written in the paper?!?!??!
            post = F.relu(self.conv5(out) + presqu, inplace=True)  # in:1/16 out:1/8
        else:
            post = F.relu(self.conv5(out) + pre, inplace=True)

        out = self.conv6(post)  # in:1/8 out:1/4

        return out, pre, post


class hourglass_2d(nn.Module):
    def __init__(self, inplanes):
        super(hourglass_2d, self).__init__()

        self.conv1 = nn.Sequential(convbn(inplanes, inplanes * 2, kernel_size=3, stride=2, pad=1, dilation=1),
                                   nn.ReLU(inplace=True))

        self.conv2 = convbn(inplanes * 2, inplanes * 2, kernel_size=3, stride=1, pad=1, dilation=1)

        self.conv3 = nn.Sequential(convbn(inplanes * 2, inplanes * 2, kernel_size=3, stride=2, pad=1, dilation=1),
                                   nn.ReLU(inplace=True))

        self.conv4 = nn.Sequential(convbn(inplanes * 2, inplanes * 2, kernel_size=3, stride=1, pad=1, dilation=1),
                                   nn.ReLU(inplace=True))

        # note: the conv5 and conv6 is without relu
        self.conv5 = nn.Sequential(
            nn.ConvTranspose2d(inplanes * 2, inplanes * 2, kernel_size=(3, 3), padding=(1, 1), output_padding=(1, 1),
                               stride=(2, 2), bias=False),
            nn.BatchNorm2d(inplanes * 2)).cuda()  # +conv2

        self.conv6 = nn.Sequential(
            nn.ConvTranspose2d(inplanes * 2, inplanes, kernel_size=(3, 3), padding=(1, 1), output_padding=(1, 1),
                               stride=(2, 2), bias=False),
            nn.BatchNorm2d(inplanes)).cuda()  # +x

    def forward(self, x, presqu, postsqu):
        out = self.conv1(x)  # in:1/4 out:1/8
        pre = self.conv2(out)  # in:1/8 out:1/8
        if postsqu is not None:
            pre = F.relu(pre + postsqu, inplace=True)  # the red connection in the figure of paper
        else:
            pre = F.relu(pre, inplace=True)

        out = self.conv3(pre)  # in:1/8 out:1/16
        out = self.conv4(out)  # in:1/16 out:1/16
        if presqu is not None:
            # the green connection
            # if this is not the first hourglass, take the output of pre-conv5 to make the fusion
            # a little different from what is written in the paper?!?!??!
            post = F.relu(self.conv5(out) + presqu, inplace=True)  # in:1/16 out:1/8
        else:
            post = F.relu(self.conv5(out) + pre, inplace=True)

        out = self.conv6(post)  # in:1/8 out:1/4

        return out, pre, post


@DIS_EMBED_BRANCHES_REGISTRY.register()
class JointEstimationDisEmbedHead(DeepLabV3PlusHead):
    """
    A semantic segmentation head of joint estimation architectures`.
    """

    @configurable
    def __init__(
            self,
            input_shape: Dict[str, ShapeSpec],
            *,
            decoder_channels: List[int],
            norm: Union[str, Callable],
            head_channels: int,
            loss_weight: float,  # the weight for the entire section
            loss_type: str,
            ignore_value: int,
            img_size=None,
            max_disp: int,
            hourglass_loss_weight: List[float],
            internal_loss_weight: List[float],
            guided_loss_weight: List[float],
            streshold_guided_loss: float,
            regression_inplanes: int,
            hourglass_inplanes: int,
            hourglass_type: str,
            resol_disp_adapt: bool,
            **kwargs,
    ):
        """
        NOTE: this interface is experimental.

        Args:
            input_shape (ShapeSpec): shape of the input feature
            decoder_channels (list[int]): a list of output channels of each
                decoder stage. It should have the same length as "input_shape"
                (each element in "input_shape" corresponds to one decoder stage).
            norm (str or callable): normalization for all conv layers.
            head_channels (int): the output channels of extra convolutions
                between decoder and predictor.
            loss_weight (float): loss weight.
            loss_top_k: (float): setting the top k% hardest pixels for
                "hard_pixel_mining" loss.
            loss_type, ignore_value, num_classes: the same as the base class.
        """
        super().__init__(
            input_shape,
            decoder_channels=decoder_channels,
            norm=norm,
            ignore_value=ignore_value,
            **kwargs,
        )

        self.loss_weight = loss_weight
        self.hourglass_loss_weight = hourglass_loss_weight
        self.internal_loss_weight = internal_loss_weight
        self.guided_loss_weight = guided_loss_weight
        self.max_disp = max_disp
        self.lamda = streshold_guided_loss
        self.loss_type = loss_type
        self.hourglass_type = hourglass_type
        self.resol_disp_adapt = resol_disp_adapt
        use_bias = norm == ""
        # `head` is additional transform before predictor
        if self.use_depthwise_separable_conv:
            # We use a single 5x5 DepthwiseSeparableConv2d to replace
            # 2 3x3 Conv2d since they have the same receptive field.
            self.head = DepthwiseSeparableConv2d(
                decoder_channels[0],
                head_channels,
                kernel_size=5,
                padding=2,
                norm1=norm,
                activation1=F.relu,
                norm2=norm,
                activation2=F.relu,
            )
        else:
            self.head = nn.Sequential(
                Conv2d(
                    decoder_channels[0],
                    decoder_channels[0],
                    kernel_size=3,
                    padding=1,
                    bias=use_bias,
                    norm=get_norm(norm, decoder_channels[0]),
                    activation=F.relu,
                ),
                Conv2d(
                    decoder_channels[0],
                    head_channels,
                    kernel_size=3,
                    padding=1,
                    bias=use_bias,
                    norm=get_norm(norm, head_channels),
                    activation=F.relu,
                ),
            )
            weight_init.c2_xavier_fill(self.head[0])
            weight_init.c2_xavier_fill(self.head[1])

        if img_size is None:
            self.img_size = [1024, 2048]  # h, w

        self.dres0 = {}
        self.dres1 = {}
        self.dres2 = {}
        self.dres3 = {}
        self.dres4 = {}
        self.classif1 = {}
        self.classif2 = {}
        self.classif3 = {}

        if self.hourglass_type == "hourglass_2D":
            zoom = [16, 8, 4]
            for i, scale in enumerate(['1/16', '1/8', '1/4']):
                if self.resol_disp_adapt:
                    max_dis = self.max_disp // zoom[i]
                else:
                    max_dis = self.max_disp // 4

                self.dres0[scale] = nn.Sequential(convbn(max_dis, hourglass_inplanes, 3, 1, 1, 1),
                                                  nn.ReLU(inplace=True),
                                                  convbn(hourglass_inplanes, hourglass_inplanes, 3, 1, 1, 1),
                                                  nn.ReLU(inplace=True))
                '''
                self.dres0[scale] = nn.Sequential(Conv2d(max_dis,
                                                         hourglass_inplanes,
                                                         kernel_size=3,
                                                         stride=1,
                                                         padding=1,
                                                         bias=use_bias,
                                                         norm=get_norm(norm, decoder_channels[idx]),
                                                         activation=F.relu,
                                                         ),

                                                  nn.Conv2d(max_dis, hourglass_inplanes, kernel_size=3, stride=1,
                                                            padding=1, dilation=0, bias=False),
                                                  nn.BatchNorm2d(hourglass_inplanes),
                                                  nn.ReLU(inplace=True),
                                                  nn.Conv2d(hourglass_inplanes, hourglass_inplanes, kernel_size=3,
                                                            stride=1,
                                                            padding=1, dilation=0, bias=False),
                                                  nn.BatchNorm2d(hourglass_inplanes),
                                                  nn.ReLU(inplace=True))
                '''
                self.dres1[scale] = nn.Sequential(convbn(hourglass_inplanes, hourglass_inplanes, 3, 1, 1, 1),
                                                  nn.ReLU(inplace=True),
                                                  convbn(hourglass_inplanes, hourglass_inplanes, 3, 1, 1, 1))
                self.dres2[scale] = hourglass_2d(hourglass_inplanes)
                self.dres3[scale] = hourglass_2d(hourglass_inplanes)
                self.dres4[scale] = hourglass_2d(hourglass_inplanes)

                self.classif1[scale] = nn.Sequential(convbn(hourglass_inplanes, hourglass_inplanes, 3, 1, 1, 1),
                                                     nn.ReLU(inplace=True),
                                                     nn.Conv2d(hourglass_inplanes, 1, kernel_size=3, padding=1,
                                                               stride=1,
                                                               bias=False)).cuda()
                self.classif2[scale] = nn.Sequential(convbn(hourglass_inplanes, hourglass_inplanes, 3, 1, 1, 1),
                                                     nn.ReLU(inplace=True),
                                                     nn.Conv2d(hourglass_inplanes, 1, kernel_size=3, padding=1,
                                                               stride=1,
                                                               bias=False))
                self.classif3[scale] = nn.Sequential(convbn(hourglass_inplanes, hourglass_inplanes, 3, 1, 1, 1),
                                                     nn.ReLU(inplace=True),
                                                     nn.Conv2d(hourglass_inplanes, 1, kernel_size=3, padding=1,
                                                               stride=1,
                                                               bias=False))
        else:
            raise ValueError("Unexpected hourglass type: %s" % self.hourglass_type)

    @classmethod
    def from_config(cls, cfg, input_shape):
        ret = super().from_config(cfg, input_shape)
        ret["head_channels"] = cfg.MODEL.DIS_EMBED_HEAD.HEAD_CHANNELS
        ret["max_disp"] = cfg.MODEL.DIS_EMBED_HEAD.MAX_DISP
        ret["hourglass_loss_weight"] = cfg.MODEL.DIS_EMBED_HEAD.HOURGLASS_LOSS_WEIGHT
        ret["internal_loss_weight"] = cfg.MODEL.DIS_EMBED_HEAD.INTERNAL_LOSS_WEIGHT
        ret["guided_loss_weight"] = cfg.MODEL.DIS_EMBED_HEAD.GUIDED_LOSS_WEIGHT
        ret["streshold_guided_loss"] = cfg.MODEL.DIS_EMBED_HEAD.STRESHOLD_GUIDED_LOSS
        ret["regression_inplanes"] = cfg.MODEL.DIS_EMBED_HEAD.REGRESSION_INPLANES
        ret["hourglass_inplanes"] = cfg.MODEL.DIS_EMBED_HEAD.HOURGLASS_INPLANES
        ret["hourglass_type"] = cfg.MODEL.DIS_EMBED_HEAD.HOURGLASS_TYPE
        ret["resol_disp_adapt"] = cfg.MODEL.DIS_EMBED_HEAD.RESOL_DISP_ADAPT
        return ret

    def forward(self, features, right_features, pyramid_features, dis_targets=None, weights=None, pan_targets=None):
        """
        Returns:
            In training, returns (None, dict of losses)
            In inference, returns (CxHxW logits, {})
        """
        y, out_features = self.layers(features)
        right_y, right_out_features = self.layers(right_features)

        for key in out_features:
            pyramid_features[key] = [[out_features[key], right_out_features[key]]]

        disparity = []  # form coarse to fine
        zoom = [16, 8, 4]
        for i, scale in enumerate(['1/16', '1/8', '1/4']):
            if self.resol_disp_adapt:
                max_dis = self.max_disp // zoom[i]
            else:
                max_dis = self.max_disp // 4
            '''
            if not len(disparity):
                dis_cost_volume = build_correlation_cost_volume(
                    max_dis, pyramid_features[scale][0][0], pyramid_features[scale][0][1])
            else:
                dis = disparity[-1][-1]
                dis_cost_volume = build_correlation_cost_volume(
                    max_dis,
                    warping(dis, pyramid_features[scale][2][0]),
                    warping(dis, pyramid_features[scale][2][1]))
            '''
            dis_cost_volume = build_correlation_cost_volume(
                max_dis, pyramid_features[scale][0][0], pyramid_features[scale][0][1])
            cost_volume = dis_cost_volume
            print(cost_volume.size())
            cost0 = self.dres0[scale](cost_volume)
            print(cost0.size())
            cost0 = self.dres1[scale](cost0) + cost0
            print(cost0.size())
            out1, pre1, post1 = self.dres2[scale](cost0, None, None)
            print(out1.size())
            out1 = out1 + cost0
            print("out1.size(): ", out1.size())
            out2, pre2, post2 = self.dres3[scale](out1, pre1, post1)
            out2 = out2 + cost0
            out3, pre3, post3 = self.dres4[scale](out2, pre1, post2)
            out3 = out3 + cost0

            print(next(self.classif1[scale].parameters()).is_cuda)
            cost1 = self.classif1[scale](out1)
            print("cost1.size(): ", cost1.size())

            print(next(self.classif2[scale].parameters()).is_cuda)
            cost2 = self.classif2[scale](out2) + cost1
            print(cost2.size())
            cost3 = self.classif3[scale](out3) + cost2
            print(cost3.size())
            raise RuntimeError('excepted stop')

            if self.training:
                cost1 = F.upsample(cost1, [max_dis, self.img_size[0], self.img_size[1]], mode='trilinear')
                cost1 = torch.squeeze(cost1, 1)
                pred1 = F.softmax(cost1, dim=1)
                pred1 = disparityregression(max_dis)(pred1)

                cost2 = F.upsample(cost2, [max_dis, self.img_size[0], self.img_size[1]], mode='trilinear')
                cost2 = torch.squeeze(cost2, 1)
                pred2 = F.softmax(cost2, dim=1)
                pred2 = disparityregression(max_dis)(pred2)

            cost3 = F.upsample(cost3, [max_dis, self.img_size[0], self.img_size[1]], mode='trilinear')
            cost3 = torch.squeeze(cost3, 1)
            pred3 = F.softmax(cost3, dim=1)
            # For your information: This formulation 'softmax(c)' learned "similarity"
            # while 'softmax(-c)' learned 'matching cost' as mentioned in the paper.
            # However, 'c' or '-c' do not affect the performance because feature-based cost volume provided flexibility.
            pred3 = disparityregression(max_dis)(pred3)  # TODO: to determine the size

            if self.training:
                if not len(disparity):
                    disparity.append([pred1, pred2, pred3])
                else:
                    disparity.append([pred1 + dis, pred2 + dis, pred3 + dis])
            else:
                if not len(disparity):
                    disparity.append([pred3])
                else:
                    disparity.append([pred3 + dis])

        if self.training:
            return self.losses(disparity, dis_targets, weights, pan_targets), None  # TODO: to be adapted
        else:
            return {}, disparity

    def layers(self, features):
        out_features = {}
        # Reverse feature maps into top-down order (from low to high resolution)
        for i, f in enumerate(self.in_features[::-1]):
            x = features[f]  # "features" is dictionary
            proj_x = self.decoder[f]["project_conv"](x)
            if self.decoder[f]["fuse_conv"] is None:
                # This is aspp module
                y = proj_x
            else:
                # Upsample y
                y = F.interpolate(y, size=proj_x.size()[2:], mode="bilinear", align_corners=False)
                y = torch.cat([proj_x, y], dim=1)
                y = self.decoder[f]["fuse_conv"](y)

            # save outputs
            if i == 1:
                out_features['1/8'] = y
            elif i == 2:
                out_features['1/4'] = y
            elif i == 0:
                out_features['1/16'] = y
            else:
                raise ValueError("undefined output of SemSeg Branch")

        y = out_features['1/4']

        return y, out_features

    def losses(self, predictions, dis_targets, weights=None, pan_targets=None):
        '''
        predictions = F.interpolate(
            predictions, scale_factor=self.common_stride, mode="bilinear", align_corners=False
        )
        '''
        mask = dis_targets < self.max_disp  # TODO: find out the effect of mask
        mask.detach_()
        loss = None
        if self.loss_type == "panoptic_guided":
            smooth_l1 = 0.0
            for i in range(len(predictions)):
                smooth_l1 = smooth_l1 + self.internal_loss_weight[i] * \
                            (self.hourglass_loss_weight[0] *
                             F.smooth_l1_loss(predictions[i][0][mask], dis_targets[mask], size_average=True) +
                             self.hourglass_loss_weight[1] *
                             F.smooth_l1_loss(predictions[i][1][mask], dis_targets[mask], size_average=True) +
                             self.hourglass_loss_weight[2] *
                             F.smooth_l1_loss(predictions[i][2][mask], dis_targets[mask]))

            pan_2rd_gradiant = cv.Laplacian(pan_targets, cv.CV_32F)
            pan_2rd_gradiant = cv.convertScaleAbs(pan_2rd_gradiant)

            bdry_loss = 0.0
            # TODO: implement the boundary loss
            for i in range(len(predictions)):
                dis_2rd_gradiant = cv.Laplacian(predictions[i][-1][mask], cv.CV_32F)
                dis_2rd_gradiant = cv.convertScaleAbs(dis_2rd_gradiant)

                print(dis_2rd_gradiant.size())
                assert dis_2rd_gradiant.size() == pan_2rd_gradiant.size()
                bdry_sum = 0.0
                count = 0
                # all pixel in the map
                for j in range(dis_2rd_gradiant.size()[0]):
                    for k in range(dis_2rd_gradiant.size()[1]):
                        # TODO: add decision
                        # if pan_2rd_gradiant[j, k] not in [road, sidewalk, vegetation, terrain]
                        count = count + 1
                        bdry_sum = bdry_sum + math.exp(-abs(dis_2rd_gradiant[j, k])) * abs(pan_2rd_gradiant[j, k])
                bdry_loss = bdry_loss + self.internal_loss_weight[i] * bdry_sum / count

            sm_loss = 0.0
            # TODO: implement the smooth loss
            for i in range(len(predictions)):
                dis_2rd_gradiant = cv.Laplacian(predictions[i][-1][mask], cv.CV_32F)
                dis_2rd_gradiant = cv.convertScaleAbs(dis_2rd_gradiant)

                sm_sum = 0.0
                count = 0
                # all pixel in the map
                for j in range(dis_2rd_gradiant.size()[0]):
                    for k in range(dis_2rd_gradiant.size()[1]):
                        # TODO: adapt decision
                        if dis_2rd_gradiant[j, k] < self.lamda:
                            count = count + 1
                            sm_sum = sm_sum + math.exp(-abs(pan_2rd_gradiant[j, k])) * abs(dis_2rd_gradiant[j, k])
                sm_loss = sm_loss + self.internal_loss_weight[i] * sm_sum / count

            loss = self.guided_loss_weight[0] * sm_loss + self.guided_loss_weight[1] * bdry_loss + \
                   self.guided_loss_weight[2] * smooth_l1

        elif self.loss_type == "smoothL1_only":
            for i in range(len(predictions)):
                loss = loss + self.internal_loss_weight[i] * \
                       (self.hourglass_loss_weight[0] *
                        F.smooth_l1_loss(predictions[i][0][mask], dis_targets[mask], size_average=True) +
                        self.hourglass_loss_weight[1] *
                        F.smooth_l1_loss(predictions[i][1][mask], dis_targets[mask], size_average=True) +
                        self.hourglass_loss_weight[2] *
                        F.smooth_l1_loss(predictions[i][2][mask], dis_targets[mask]))
        else:
            raise ValueError("Unexpected loss type: %s" % self.loss_type)

        losses = {"loss_dis": loss * self.loss_weight}
        return losses
