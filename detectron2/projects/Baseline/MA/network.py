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

from typing import Optional, Tuple
from detectron2.layers import ASPP

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
from torchsummary import summary

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

        self.max_disp = cfg.MODEL.DIS_EMBED_HEAD.MAX_DISP

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

        # dict{'1/4': [[left_seg, right_seg], [left_ins, right_ins], [left_dis, right_dis]], ...}
        pyramid_features = {}
        for key in left_sem_seg_features:
            pyramid_features[key] = []
            pyramid_features[key].append([left_sem_seg_features[key], right_sem_seg_features[key]])
            pyramid_features[key].append([left_ins_seg_features[key], right_ins_seg_features[key]])

        dis_targets = [x["dis_est"].to(self.device) for x in batched_inputs]
        dis_targets = ImageList.from_tensors(dis_targets, size_divisibility).tensor
        dis_targets.detach_()
        dis_mask = [x["dis_mask"].to(self.device) for x in batched_inputs]
        dis_mask = ImageList.from_tensors(dis_mask, size_divisibility).tensor
        dis_mask.detach_()

        pan_guided = [x["pan_gui"].to(self.device) for x in batched_inputs]
        pan_guided = ImageList.from_tensors(pan_guided, size_divisibility).tensor
        pan_guided.detach_()
        pan_mask = [x["pan_mask"].to(self.device) for x in batched_inputs]
        pan_mask = ImageList.from_tensors(pan_mask, size_divisibility).tensor
        pan_mask.detach_()

        dis_embed_loss, dis_result = self.dis_embed_head(left_features, right_features, pyramid_features,
                                                         dis_targets=dis_targets,
                                                         dis_mask=dis_mask, pan_guided=pan_guided, pan_mask=pan_mask)
        losses.update(dis_embed_loss)

        if self.training:
            return losses
        if self.benchmark_network_speed:
            return []

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
            img_size: List[int],
            max_disp: int,
            hourglass_loss_weight: List[float],
            internal_loss_weight: List[float],
            guided_loss_weight: List[float],
            streshold_guided_loss: float,
            regression_inplanes: int,
            hourglass_inplanes: int,
            hourglass_type: str,
            resol_disp_adapt: bool,
            gradient_type: str,
            # num_classes=None,
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
        self.hourglass_loss_weight = hourglass_loss_weight
        self.internal_loss_weight = internal_loss_weight
        self.guided_loss_weight = guided_loss_weight
        self.max_disp = max_disp
        self.lamda = streshold_guided_loss
        self.loss_type = loss_type
        self.hourglass_type = hourglass_type
        self.resol_disp_adapt = resol_disp_adapt
        self.gradient_type = gradient_type
        self.loss = None
        self.predictor = None
        self.predictor = nn.ModuleDict()

        if img_size is None:
            self.img_size = [1024, 2048]  # h, w
        else:
            self.img_size = img_size

        self.warp = Warper2d(direction_str='r2l', pad_mode="zeros")

        if self.hourglass_type == "hourglass_2D":
            zoom = [16, 8, 4]
            for i, scale in enumerate(['1/16', '1/8', '1/4']):
                decoder_stage = nn.ModuleDict()
                if self.resol_disp_adapt:
                    max_dis = self.max_disp // zoom[i]
                else:
                    max_dis = self.max_disp

                dres0 = nn.Sequential(convbn(max_dis, max_dis, 3, 1, 1, 1),
                                      nn.ReLU(inplace=True),
                                      convbn(max_dis, max_dis, 3, 1, 1, 1),
                                      nn.ReLU(inplace=True))
                decoder_stage['dres0'] = dres0
                dres1 = nn.Sequential(convbn(max_dis, max_dis, 3, 1, 1, 1),
                                      nn.ReLU(inplace=True),
                                      convbn(max_dis, max_dis, 3, 1, 1, 1))
                decoder_stage['dres1'] = dres1
                hourglass_inplanes = max_dis
                dres2 = hourglass_2d(hourglass_inplanes)
                dres3 = hourglass_2d(hourglass_inplanes)
                dres4 = hourglass_2d(hourglass_inplanes)
                decoder_stage['dres2'] = dres2
                decoder_stage['dres3'] = dres3
                decoder_stage['dres4'] = dres4

                classif1 = nn.Sequential(convbn(hourglass_inplanes, hourglass_inplanes, 3, 1, 1, 1),
                                         nn.ReLU(inplace=True),
                                         nn.Conv2d(hourglass_inplanes, hourglass_inplanes, kernel_size=3,
                                                   padding=1,
                                                   stride=1,
                                                   bias=False)).cuda()
                decoder_stage['classif1'] = classif1
                classif2 = nn.Sequential(convbn(hourglass_inplanes, hourglass_inplanes, 3, 1, 1, 1),
                                         nn.ReLU(inplace=True),
                                         nn.Conv2d(hourglass_inplanes, hourglass_inplanes, kernel_size=3,
                                                   padding=1,
                                                   stride=1,
                                                   bias=False)).cuda()
                decoder_stage['classif2'] = classif2
                classif3 = nn.Sequential(convbn(hourglass_inplanes, hourglass_inplanes, 3, 1, 1, 1),
                                         nn.ReLU(inplace=True),
                                         nn.Conv2d(hourglass_inplanes, hourglass_inplanes, kernel_size=3,
                                                   padding=1,
                                                   stride=1,
                                                   bias=False)).cuda()
                decoder_stage['classif3'] = classif3
                self.predictor[scale] = decoder_stage
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
        ret["loss_type"] = cfg.MODEL.DIS_EMBED_HEAD.LOSS_TYPE
        ret["gradient_type"] = cfg.MODEL.DIS_EMBED_HEAD.GRADIENT_TYPE
        ret["img_size"] = cfg.INPUT.IMG_SIZE
        ret["num_classes"] = cfg.MODEL.DIS_EMBED_HEAD.NUM_CLASSES
        return ret

    def forward(self, features, right_features, pyramid_features, dis_targets=None, dis_mask=None, weights=None,
                pan_guided=None, pan_mask=None, ):
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
                max_dis = self.max_disp
            if not len(disparity):
                seg_cost_volume = build_correlation_cost_volume(
                    max_dis, pyramid_features[scale][0][0], pyramid_features[scale][0][1])
                ins_cost_volume = build_correlation_cost_volume(
                    max_dis, pyramid_features[scale][1][0], pyramid_features[scale][1][1])
                dis_cost_volume = build_correlation_cost_volume(
                    max_dis, pyramid_features[scale][2][0], pyramid_features[scale][2][1])
            else:
                dis = disparity[-1][-1]
                seg_cost_volume = build_correlation_cost_volume(
                    max_dis,
                    pyramid_features[scale][0][0],
                    self.warp(dis, pyramid_features[scale][0][1], scale), )
                ins_cost_volume = build_correlation_cost_volume(
                    max_dis,
                    pyramid_features[scale][1][0],
                    self.warp(dis, pyramid_features[scale][1][1], scale))
                dis_cost_volume = build_correlation_cost_volume(
                    max_dis,
                    pyramid_features[scale][2][0],
                    self.warp(dis, pyramid_features[scale][2][1], scale))
            cost_volume = seg_cost_volume * ins_cost_volume * dis_cost_volume

            cost0 = self.predictor[scale]['dres0'](cost_volume)
            cost0 = self.predictor[scale]['dres1'](cost0) + cost0
            out1, pre1, post1 = self.predictor[scale]['dres2'](cost0, None, None)
            out1 = out1 + cost0
            out2, pre2, post2 = self.predictor[scale]['dres3'](out1, pre1, post1)
            out2 = out2 + cost0
            out3, pre3, post3 = self.predictor[scale]['dres4'](out2, pre1, post2)
            out3 = out3 + cost0
            cost1 = self.predictor[scale]['classif1'](out1)
            cost2 = self.predictor[scale]['classif2'](out2) + cost1
            cost3 = self.predictor[scale]['classif3'](out3) + cost2

            if self.training:
                cost1 = torch.unsqueeze(cost1, 1)
                cost1 = F.interpolate(cost1, size=[max_dis, self.img_size[0], self.img_size[1]], mode='trilinear',
                                      align_corners=True)
                cost1 = torch.squeeze(cost1, 1)
                pred1 = F.softmax(cost1, dim=1)
                pred1 = disparityregression(max_dis)(pred1)

                cost2 = torch.unsqueeze(cost2, 1)
                cost2 = F.interpolate(cost2, size=[max_dis, self.img_size[0], self.img_size[1]], mode='trilinear',
                                      align_corners=True)
                cost2 = torch.squeeze(cost2, 1)
                pred2 = F.softmax(cost2, dim=1)
                pred2 = disparityregression(max_dis)(pred2)

            cost3 = torch.unsqueeze(cost3, 1)
            cost3 = F.interpolate(cost3, size=[max_dis, self.img_size[0], self.img_size[1]], mode='trilinear',
                                  align_corners=True)
            cost3 = torch.squeeze(cost3, 1)
            pred3 = F.softmax(cost3, dim=1)
            pred3 = disparityregression(max_dis)(pred3)  # TODO: to determine the size

            if self.training:
                if not len(disparity):
                    disparity.append([pred1, pred2, pred3])  # List[3x List(3x Tensor)]
                else:
                    disparity.append([pred1 + dis, pred2 + dis, pred3 + dis])
            else:
                if not len(disparity):
                    disparity.append([pred3])
                else:
                    disparity.append([pred3 + dis])

        if self.training:
            return self.losses(disparity, dis_targets=dis_targets, dis_mask=dis_mask, weights=weights,
                               pan_guided=pan_guided, pan_mask=pan_mask), disparity
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

    def losses(self, predictions, dis_targets=None, dis_mask=None, weights=None,
               pan_guided=None, pan_mask=None):

        dis_mask = torch.unsqueeze(dis_mask, 1)
        dis_targets = torch.unsqueeze(dis_targets, 1)
        dis_mask_bool = dis_mask == 1.0
        dis_mask_bool.detach_()

        if self.loss_type == "panoptic_guided":
            get_gradient = Gradient(self.gradient_type)

            # prepare the panoptic guided ground truth
            pan_guided_target = torch.unsqueeze(pan_guided, 1)
            pan_gradiant_x, pan_gradiant_y = get_gradient(pan_guided_target)
            pan_gradiant_x = pan_gradiant_x.detach_()
            pan_gradiant_y = pan_gradiant_y.detach_()
            pan_mask = torch.unsqueeze(pan_mask, 1)
            pan_mask = pan_mask[:, :, 1:-1, 1:-1]  # to adapt the changes after gradient
            pan_mask_bool = pan_mask == 1.0
            pan_mask_bool.detach_()

            bdry_loss = None
            sm_loss = None
            for i in range(len(predictions)):  # for each pyramid
                bdry_loss_pyramid = None
                sm_loss_pyramid = None
                for j in range(len(predictions[0])):  # for each stage of hourglass
                    # get gradient of predictions
                    pred_guided_gradiant_x, pred_guided_gradiant_y = get_gradient(predictions[i][j])
                    assert pan_gradiant_x.shape == pred_guided_gradiant_x.shape
                    assert pan_gradiant_y.shape == pred_guided_gradiant_y.shape

                    # get bdry_loss_pyramid
                    bdry_sum = (torch.exp(-pred_guided_gradiant_x[pan_mask_bool]).mul(pan_gradiant_x[pan_mask_bool]) +
                                torch.exp(-pred_guided_gradiant_y[pan_mask_bool]).mul(pan_gradiant_y[pan_mask_bool]))
                    if bdry_loss_pyramid:
                        bdry_loss_pyramid = self.hourglass_loss_weight[j] * torch.mean(bdry_sum) + bdry_loss_pyramid
                    else:
                        bdry_loss_pyramid = self.hourglass_loss_weight[j] * torch.mean(bdry_sum)

                    # get sm_loss_pyramid
                    sm_mask_x = pred_guided_gradiant_x < self.lamda
                    sm_mask_y = pred_guided_gradiant_y < self.lamda
                    sm_mask = sm_mask_x & sm_mask_y
                    sm_mask.detach_()
                    sm_sum = (torch.exp(-pan_gradiant_x[sm_mask]).mul(pred_guided_gradiant_x[sm_mask]) +
                              torch.exp(-pan_gradiant_y[sm_mask]).mul(pred_guided_gradiant_y[sm_mask]))
                    if sm_loss_pyramid:
                        sm_loss_pyramid = self.hourglass_loss_weight[j] * torch.mean(sm_sum) + sm_loss_pyramid
                    else:
                        sm_loss_pyramid = self.hourglass_loss_weight[j] * torch.mean(sm_sum)
                assert bdry_loss_pyramid
                assert sm_loss_pyramid

                if bdry_loss:
                    bdry_loss = self.internal_loss_weight[i] * bdry_loss_pyramid + bdry_loss
                else:
                    bdry_loss = self.internal_loss_weight[i] * bdry_loss_pyramid

                if sm_loss:
                    sm_loss = self.internal_loss_weight[i] * sm_loss_pyramid + sm_loss
                else:
                    sm_loss = self.internal_loss_weight[i] * sm_loss_pyramid
            assert bdry_loss
            assert sm_loss

            smooth_l1 = None
            for i in range(len(predictions)):  # for each pyramid
                if smooth_l1:
                    smooth_l1 = smooth_l1 + self.internal_loss_weight[i] * \
                                (self.hourglass_loss_weight[0] *
                                 F.smooth_l1_loss(predictions[i][0][dis_mask_bool], dis_targets[dis_mask_bool],
                                                  reduction='mean') +
                                 self.hourglass_loss_weight[1] *
                                 F.smooth_l1_loss(predictions[i][1][dis_mask_bool], dis_targets[dis_mask_bool],
                                                  reduction='mean') +
                                 self.hourglass_loss_weight[2] *
                                 F.smooth_l1_loss(predictions[i][2][dis_mask_bool], dis_targets[dis_mask_bool]))
                else:
                    smooth_l1 = self.internal_loss_weight[i] * \
                                (self.hourglass_loss_weight[0] *
                                 F.smooth_l1_loss(predictions[i][0][dis_mask_bool], dis_targets[dis_mask_bool],
                                                  reduction='mean') +
                                 self.hourglass_loss_weight[1] *
                                 F.smooth_l1_loss(predictions[i][1][dis_mask_bool], dis_targets[dis_mask_bool],
                                                  reduction='mean') +
                                 self.hourglass_loss_weight[2] *
                                 F.smooth_l1_loss(predictions[i][2][dis_mask_bool], dis_targets[dis_mask_bool]))
            assert smooth_l1

            loss = self.guided_loss_weight[0] * sm_loss + self.guided_loss_weight[1] * bdry_loss + \
                   self.guided_loss_weight[2] * smooth_l1

        elif self.loss_type == "smoothL1_only":
            smooth_l1 = None
            for i in range(len(predictions)):  # for each pyramid
                if smooth_l1:
                    smooth_l1 = smooth_l1 + self.internal_loss_weight[i] * \
                                (self.hourglass_loss_weight[0] *
                                 F.smooth_l1_loss(predictions[i][0][dis_mask_bool], dis_targets[dis_mask_bool],
                                                  reduction='mean') +
                                 self.hourglass_loss_weight[1] *
                                 F.smooth_l1_loss(predictions[i][1][dis_mask_bool], dis_targets[dis_mask_bool],
                                                  reduction='mean') +
                                 self.hourglass_loss_weight[2] *
                                 F.smooth_l1_loss(predictions[i][2][dis_mask_bool], dis_targets[dis_mask_bool]))
                else:
                    smooth_l1 = self.internal_loss_weight[i] * \
                                (self.hourglass_loss_weight[0] *
                                 F.smooth_l1_loss(predictions[i][0][dis_mask_bool], dis_targets[dis_mask_bool],
                                                  reduction='mean') +
                                 self.hourglass_loss_weight[1] *
                                 F.smooth_l1_loss(predictions[i][1][dis_mask_bool], dis_targets[dis_mask_bool],
                                                  reduction='mean') +
                                 self.hourglass_loss_weight[2] *
                                 F.smooth_l1_loss(predictions[i][2][dis_mask_bool], dis_targets[dis_mask_bool]))
            assert smooth_l1

            loss = smooth_l1

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


# TODO: transform to function
class Warper2d(nn.Module):
    def __init__(self, direction_str='r2l', pad_mode="zeros"):
        super().__init__()

        if direction_str == 'r2l':
            self.direction = - 1
        elif direction_str == 'l2r':
            self.direction = 1
        self.pad_mode = pad_mode

        self.scale = {"1/16": 0.0625,
                      "1/8": 0.125,
                      "1/4": 0.25, }

    def forward(self, disp, img, scale):
        '''

        Args:
            disp: tensor: (b,h,w)
            img: tensor: (b,256, h,w)
            scale:

        Returns:

        '''
        '''
        disp_ = disp.detach()
        img_ = img.detach()
        '''

        [B, _, H, W] = img.size()
        # mesh grid
        xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
        xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
        yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
        yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
        grid = torch.cat((xx, yy), 1).float().cuda()
        grid.detach_()
        # disp = torch.unsqueeze(disp, 1)  # (b,1, h,w)
        assert len(disp.shape) == 4
        scale_factor = self.scale[scale]
        disp = F.interpolate(disp, scale_factor=scale_factor)

        vgrid = grid + disp * self.direction
        vgrid[:, 1, :, :] = grid[:, 1, :, :]
        # scale grid to [-1,1]
        vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :].clone() / max(W - 1, 1) - 1.0
        vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :].clone() / max(H - 1, 1) - 1.0
        vgrid = vgrid.permute(0, 2, 3, 1)  # b, h, w, c

        output = F.grid_sample(img, vgrid, padding_mode=self.pad_mode, align_corners=True)

        mask = torch.ones(img.size()).cuda()
        mask.detach_()
        mask = F.grid_sample(mask, vgrid, align_corners=True)

        mask[mask < 0.9999] = 0
        mask[mask > 0] = 1

        return output * mask  # pre elements multi


def optical_flow_warping(x, flo, pad_mode="zeros"):
    """
    warp an image/tensor (im2) back to im1, according to the optical flow

    x: [B, C, H, W] (im2)
    flo: [B, 2, H, W] flow
    pad_mode (optional): ref to https://pytorch.org/docs/stable/nn.functional.html#grid-sample
        "zeros": use 0 for out-of-bound grid locations,
        "border": use border values for out-of-bound grid locations
    """
    B, C, H, W = x.size()
    # mesh grid
    xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
    yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
    xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
    yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
    grid = torch.cat((xx, yy), 1).float()

    vgrid = grid + flo  # warp后，新图每个像素对应原图的位置

    # scale grid to [-1,1]
    vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :].clone() / max(W - 1, 1) - 1.0
    vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :].clone() / max(H - 1, 1) - 1.0

    vgrid = vgrid.permute(0, 2, 3, 1)
    output = F.grid_sample(x, vgrid, padding_mode=pad_mode)

    mask = torch.ones(x.size())
    mask = F.grid_sample(mask, vgrid)

    mask[mask < 0.9999] = 0
    mask[mask > 0] = 1

    return output * mask


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
        super().__init__()

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


class Gradient(nn.Module):
    def __init__(self, grad_type="sobel"):
        self.grad_type = grad_type
        super(Gradient, self).__init__()
        if self.grad_type == "sobel":
            kernel_x = [[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]
            kernel_y = [[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]
            kernel_y = torch.FloatTensor(kernel_y).unsqueeze(0).unsqueeze(0).cuda()
            self.weight_y = nn.Parameter(data=kernel_y, requires_grad=False)
        elif self.grad_type == "laplacian":
            kernel_x = [[-1., -1., -1.], [-1., 8., -1.], [-1., -1., -1.]]
        else:
            raise ValueError("Unexpected gradient type: %s" % self.grad_type)
        kernel_x = torch.FloatTensor(kernel_x).unsqueeze(0).unsqueeze(0).cuda()
        self.weight_x = nn.Parameter(data=kernel_x, requires_grad=False)

    def forward(self, x):
        grad_x = F.conv1d(x, self.weight_x, padding="valid")
        gradient_x = torch.abs(grad_x)
        if self.grad_type == "sobel":
            grad_y = F.conv1d(x, self.weight_y, padding="valid")
            gradient_y = torch.abs(grad_y)
            return gradient_x, gradient_y
        elif self.grad_type == "laplacian":
            return gradient_x, None
        else:
            raise ValueError("Unexpected gradient type: %s" % self.grad_type)


class DeepLabV3PlusHeadDecoder(nn.Module):
    """
    A semantic segmentation head described in :paper:`DeepLabV3+`.
    """

    @configurable
    def __init__(
            self,
            input_shape: Dict[str, ShapeSpec],
            *,
            project_channels: List[int],
            aspp_dilations: List[int],
            aspp_dropout: float,
            decoder_channels: List[int],
            common_stride: int,
            norm: Union[str, Callable],
            train_size: Optional[Tuple],
            loss_weight: float = 1.0,
            loss_type: str = "cross_entropy",
            ignore_value: int = -1,
            num_classes: Optional[int] = None,
            use_depthwise_separable_conv: bool = False,
    ):
        """
        NOTE: this interface is experimental.

        Args:
            input_shape: shape of the input features. They will be ordered by stride
                and the last one (with largest stride) is used as the input to the
                decoder (i.e.  the ASPP module); the rest are low-level feature for
                the intermediate levels of decoder.
            project_channels (list[int]): a list of low-level feature channels.
                The length should be len(in_features) - 1.
            aspp_dilations (list(int)): a list of 3 dilations in ASPP.
            aspp_dropout (float): apply dropout on the output of ASPP.
            decoder_channels (list[int]): a list of output channels of each
                decoder stage. It should have the same length as "in_features"
                (each element in "in_features" corresponds to one decoder stage).
            common_stride (int): output stride of decoder.
            norm (str or callable): normalization for all conv layers.
            train_size (tuple): (height, width) of training images.
            loss_weight (float): loss weight.
            loss_type (str): type of loss function, 2 opptions:
                (1) "cross_entropy" is the standard cross entropy loss.
                (2) "hard_pixel_mining" is the loss in DeepLab that samples
                    top k% hardest pixels.
            ignore_value (int): category to be ignored during training.
            num_classes (int): number of classes, if set to None, the decoder
                will not construct a predictor.
            use_depthwise_separable_conv (bool): use DepthwiseSeparableConv2d
                in ASPP and decoder.
        """
        super().__init__()
        input_shape = sorted(input_shape.items(), key=lambda x: x[1].stride)

        # fmt: off
        self.in_features = [k for k, v in input_shape]  # starting from "res2" to "res5"
        in_channels = [x[1].channels for x in input_shape]
        in_strides = [x[1].stride for x in input_shape]
        aspp_channels = decoder_channels[-1]
        self.ignore_value = ignore_value
        self.common_stride = common_stride  # output stride
        self.loss_weight = loss_weight
        self.loss_type = loss_type
        self.decoder_only = True
        self.use_depthwise_separable_conv = use_depthwise_separable_conv
        # fmt: on

        assert (
                len(project_channels) == len(self.in_features) - 1
        ), "Expected {} project_channels, got {}".format(
            len(self.in_features) - 1, len(project_channels)
        )
        assert len(decoder_channels) == len(
            self.in_features
        ), "Expected {} decoder_channels, got {}".format(
            len(self.in_features), len(decoder_channels)
        )
        self.decoder = nn.ModuleDict()

        use_bias = norm == ""
        for idx, in_channel in enumerate(in_channels):
            decoder_stage = nn.ModuleDict()

            if idx == len(self.in_features) - 1:
                # ASPP module
                if train_size is not None:
                    train_h, train_w = train_size
                    encoder_stride = in_strides[-1]
                    if train_h % encoder_stride or train_w % encoder_stride:
                        raise ValueError("Crop size need to be divisible by encoder stride.")
                    pool_h = train_h // encoder_stride
                    pool_w = train_w // encoder_stride
                    pool_kernel_size = (pool_h, pool_w)
                else:
                    pool_kernel_size = None
                project_conv = ASPP(
                    in_channel,
                    aspp_channels,
                    aspp_dilations,
                    norm=norm,
                    activation=F.relu,
                    pool_kernel_size=pool_kernel_size,
                    dropout=aspp_dropout,
                    use_depthwise_separable_conv=use_depthwise_separable_conv,
                )
                fuse_conv = None
            else:
                project_conv = Conv2d(
                    in_channel,
                    project_channels[idx],
                    kernel_size=1,
                    bias=use_bias,
                    norm=get_norm(norm, project_channels[idx]),
                    activation=F.relu,
                )
                weight_init.c2_xavier_fill(project_conv)
                if use_depthwise_separable_conv:
                    # We use a single 5x5 DepthwiseSeparableConv2d to replace
                    # 2 3x3 Conv2d since they have the same receptive field,
                    # proposed in :paper:`Panoptic-DeepLab`.
                    fuse_conv = DepthwiseSeparableConv2d(
                        project_channels[idx] + decoder_channels[idx + 1],
                        decoder_channels[idx],
                        kernel_size=5,
                        padding=2,
                        norm1=norm,
                        activation1=F.relu,
                        norm2=norm,
                        activation2=F.relu,
                    )
                else:
                    fuse_conv = nn.Sequential(
                        Conv2d(
                            project_channels[idx] + decoder_channels[idx + 1],
                            decoder_channels[idx],
                            kernel_size=3,
                            padding=1,
                            bias=use_bias,
                            norm=get_norm(norm, decoder_channels[idx]),
                            activation=F.relu,
                        ),
                        Conv2d(
                            decoder_channels[idx],
                            decoder_channels[idx],
                            kernel_size=3,
                            padding=1,
                            bias=use_bias,
                            norm=get_norm(norm, decoder_channels[idx]),
                            activation=F.relu,
                        ),
                    )
                    weight_init.c2_xavier_fill(fuse_conv[0])
                    weight_init.c2_xavier_fill(fuse_conv[1])

            decoder_stage["project_conv"] = project_conv
            decoder_stage["fuse_conv"] = fuse_conv

            self.decoder[self.in_features[idx]] = decoder_stage

    @classmethod
    def from_config(cls, cfg, input_shape):
        if cfg.INPUT.CROP.ENABLED:
            assert cfg.INPUT.CROP.TYPE == "absolute"
            train_size = cfg.INPUT.CROP.SIZE
        else:
            train_size = None
        decoder_channels = [cfg.MODEL.SEM_SEG_HEAD.CONVS_DIM] * (
                len(cfg.MODEL.SEM_SEG_HEAD.IN_FEATURES) - 1
        ) + [cfg.MODEL.SEM_SEG_HEAD.ASPP_CHANNELS]
        ret = dict(
            input_shape={
                k: v for k, v in input_shape.items() if k in cfg.MODEL.SEM_SEG_HEAD.IN_FEATURES
            },
            project_channels=cfg.MODEL.SEM_SEG_HEAD.PROJECT_CHANNELS,
            aspp_dilations=cfg.MODEL.SEM_SEG_HEAD.ASPP_DILATIONS,
            aspp_dropout=cfg.MODEL.SEM_SEG_HEAD.ASPP_DROPOUT,
            decoder_channels=decoder_channels,
            common_stride=cfg.MODEL.SEM_SEG_HEAD.COMMON_STRIDE,
            norm=cfg.MODEL.SEM_SEG_HEAD.NORM,
            train_size=train_size,
            loss_weight=cfg.MODEL.SEM_SEG_HEAD.LOSS_WEIGHT,
            loss_type=cfg.MODEL.SEM_SEG_HEAD.LOSS_TYPE,
            ignore_value=cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE,
            num_classes=cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES,
            use_depthwise_separable_conv=cfg.MODEL.SEM_SEG_HEAD.USE_DEPTHWISE_SEPARABLE_CONV,
        )
        return ret

    def forward(self, features, targets=None):
        """
        Returns:
            In training, returns (None, dict of losses)
            In inference, returns (CxHxW logits, {})
        """
        y = self.layers(features)
        if self.decoder_only:
            # Output from self.layers() only contains decoder feature.
            return y
        if self.training:
            return None, self.losses(y, targets)
        else:
            y = F.interpolate(
                y, scale_factor=self.common_stride, mode="bilinear", align_corners=False
            )
            return y, {}

    def layers(self, features):
        # Reverse feature maps into top-down order (from low to high resolution)
        for f in self.in_features[::-1]:
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
        if not self.decoder_only:
            y = self.predictor(y)
        return y

    def losses(self, predictions, targets):
        predictions = F.interpolate(
            predictions, scale_factor=self.common_stride, mode="bilinear", align_corners=False
        )
        loss = self.loss(predictions, targets)
        losses = {"loss_sem_seg": loss * self.loss_weight}
        return losses


@DIS_EMBED_BRANCHES_REGISTRY.register()
class JointEstimationDisEmbedHead_backup(DeepLabV3PlusHead):
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
            img_size: List[int],
            max_disp: int,
            hourglass_loss_weight: List[float],
            internal_loss_weight: List[float],
            guided_loss_weight: List[float],
            streshold_guided_loss: float,
            regression_inplanes: int,
            hourglass_inplanes: int,
            hourglass_type: str,
            resol_disp_adapt: bool,
            gradient_type: str,
            # num_classes=None,
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
        self.gradient_type = gradient_type
        self.decoder_only = True
        self.loss = None
        self.predictor = None
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
        else:
            self.img_size = img_size

        self.warp = Warper2d(direction_str='r2l', pad_mode="zeros")
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
                    max_dis = self.max_disp

                self.dres0[scale] = nn.Sequential(convbn(max_dis, max_dis, 3, 1, 1, 1),
                                                  nn.ReLU(inplace=True),
                                                  convbn(max_dis, max_dis, 3, 1, 1, 1),
                                                  nn.ReLU(inplace=True))
                self.dres1[scale] = nn.Sequential(convbn(max_dis, max_dis, 3, 1, 1, 1),
                                                  nn.ReLU(inplace=True),
                                                  convbn(max_dis, max_dis, 3, 1, 1, 1))
                hourglass_inplanes = max_dis
                self.dres2[scale] = hourglass_2d(hourglass_inplanes)
                self.dres3[scale] = hourglass_2d(hourglass_inplanes)
                self.dres4[scale] = hourglass_2d(hourglass_inplanes)

                self.classif1[scale] = nn.Sequential(convbn(hourglass_inplanes, hourglass_inplanes, 3, 1, 1, 1),
                                                     nn.ReLU(inplace=True),
                                                     nn.Conv2d(hourglass_inplanes, hourglass_inplanes, kernel_size=3,
                                                               padding=1,
                                                               stride=1,
                                                               bias=False)).cuda()
                self.classif2[scale] = nn.Sequential(convbn(hourglass_inplanes, hourglass_inplanes, 3, 1, 1, 1),
                                                     nn.ReLU(inplace=True),
                                                     nn.Conv2d(hourglass_inplanes, hourglass_inplanes, kernel_size=3,
                                                               padding=1,
                                                               stride=1,
                                                               bias=False)).cuda()
                self.classif3[scale] = nn.Sequential(convbn(hourglass_inplanes, hourglass_inplanes, 3, 1, 1, 1),
                                                     nn.ReLU(inplace=True),
                                                     nn.Conv2d(hourglass_inplanes, hourglass_inplanes, kernel_size=3,
                                                               padding=1,
                                                               stride=1,
                                                               bias=False)).cuda()
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
        ret["loss_type"] = cfg.MODEL.DIS_EMBED_HEAD.LOSS_TYPE
        ret["gradient_type"] = cfg.MODEL.DIS_EMBED_HEAD.GRADIENT_TYPE
        ret["img_size"] = cfg.INPUT.IMG_SIZE
        # ret["num_classes"] = cfg.MODEL.DIS_EMBED_HEAD.NUM_CLASSES
        return ret

    def forward(self, features, right_features, pyramid_features, dis_targets=None, dis_mask=None, weights=None,
                pan_guided=None, pan_mask=None, ):
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
                max_dis = self.max_disp
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
                    pyramid_features[scale][0][0],
                    self.warp(dis, pyramid_features[scale][0][1], scale), )
                ins_cost_volume = build_correlation_cost_volume(
                    max_dis,
                    pyramid_features[scale][1][0],
                    self.warp(dis, pyramid_features[scale][1][1], scale))
                # print(seg_cost_volume)
                dis_cost_volume = build_correlation_cost_volume(
                    max_dis,
                    pyramid_features[scale][2][0],
                    self.warp(dis, pyramid_features[scale][2][1], scale))
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
                cost1 = torch.unsqueeze(cost1, 1)
                cost1 = F.interpolate(cost1, size=[max_dis, self.img_size[0], self.img_size[1]], mode='trilinear',
                                      align_corners=True)
                cost1 = torch.squeeze(cost1, 1)
                pred1 = F.softmax(cost1, dim=1)
                pred1 = disparityregression(max_dis)(pred1)

                cost2 = torch.unsqueeze(cost2, 1)
                cost2 = F.interpolate(cost2, size=[max_dis, self.img_size[0], self.img_size[1]], mode='trilinear',
                                      align_corners=True)
                cost2 = torch.squeeze(cost2, 1)
                pred2 = F.softmax(cost2, dim=1)
                pred2 = disparityregression(max_dis)(pred2)

            cost3 = torch.unsqueeze(cost3, 1)
            cost3 = F.interpolate(cost3, size=[max_dis, self.img_size[0], self.img_size[1]], mode='trilinear',
                                  align_corners=True)
            cost3 = torch.squeeze(cost3, 1)
            pred3 = F.softmax(cost3, dim=1)
            pred3 = disparityregression(max_dis)(pred3)  # TODO: to determine the size

            if self.training:
                if not len(disparity):
                    disparity.append([pred1, pred2, pred3])  # List[3x List(3x Tensor)]
                else:
                    disparity.append([pred1 + dis, pred2 + dis, pred3 + dis])
            else:
                if not len(disparity):
                    disparity.append([pred3])
                else:
                    disparity.append([pred3 + dis])

        if self.training:
            return self.losses(disparity, dis_targets=dis_targets, dis_mask=dis_mask, weights=weights,
                               pan_guided=pan_guided, pan_mask=pan_mask), disparity
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

    def losses(self, predictions, dis_targets=None, dis_mask=None, weights=None,
               pan_guided=None, pan_mask=None):

        dis_mask = torch.unsqueeze(dis_mask, 1)
        dis_targets = torch.unsqueeze(dis_targets, 1)
        dis_mask_bool = dis_mask == 1.0
        dis_mask_bool.detach_()

        if self.loss_type == "panoptic_guided":
            get_gradient = Gradient(self.gradient_type)

            # prepare the panoptic guided ground truth
            pan_guided_target = torch.unsqueeze(pan_guided, 1)
            pan_gradiant_x, pan_gradiant_y = get_gradient(pan_guided_target)
            pan_gradiant_x = pan_gradiant_x.detach_()
            pan_gradiant_y = pan_gradiant_y.detach_()
            pan_mask = torch.unsqueeze(pan_mask, 1)
            pan_mask = pan_mask[:, :, 1:-1, 1:-1]  # to adapt the changes after gradient
            pan_mask_bool = pan_mask == 1.0
            pan_mask_bool.detach_()

            bdry_loss = None
            sm_loss = None
            for i in range(len(predictions)):  # for each pyramid
                bdry_loss_pyramid = None
                sm_loss_pyramid = None
                for j in range(len(predictions[0])):  # for each stage of hourglass
                    # get gradient of predictions
                    pred_guided_gradiant_x, pred_guided_gradiant_y = get_gradient(predictions[i][j])
                    assert pan_gradiant_x.shape == pred_guided_gradiant_x.shape
                    assert pan_gradiant_y.shape == pred_guided_gradiant_y.shape

                    # get bdry_loss_pyramid
                    bdry_sum = (torch.exp(-pred_guided_gradiant_x[pan_mask_bool]).mul(pan_gradiant_x[pan_mask_bool]) +
                                torch.exp(-pred_guided_gradiant_y[pan_mask_bool]).mul(pan_gradiant_y[pan_mask_bool]))
                    if bdry_loss_pyramid:
                        bdry_loss_pyramid = self.hourglass_loss_weight[j] * torch.mean(bdry_sum) + bdry_loss_pyramid
                    else:
                        bdry_loss_pyramid = self.hourglass_loss_weight[j] * torch.mean(bdry_sum)

                    # get sm_loss_pyramid
                    sm_mask_x = pred_guided_gradiant_x < self.lamda
                    sm_mask_y = pred_guided_gradiant_y < self.lamda
                    sm_mask = sm_mask_x & sm_mask_y
                    sm_mask.detach_()
                    sm_sum = (torch.exp(-pan_gradiant_x[sm_mask]).mul(pred_guided_gradiant_x[sm_mask]) +
                              torch.exp(-pan_gradiant_y[sm_mask]).mul(pred_guided_gradiant_y[sm_mask]))
                    if sm_loss_pyramid:
                        sm_loss_pyramid = self.hourglass_loss_weight[j] * torch.mean(sm_sum) + sm_loss_pyramid
                    else:
                        sm_loss_pyramid = self.hourglass_loss_weight[j] * torch.mean(sm_sum)
                assert bdry_loss_pyramid
                assert sm_loss_pyramid

                if bdry_loss:
                    bdry_loss = self.internal_loss_weight[i] * bdry_loss_pyramid + bdry_loss
                else:
                    bdry_loss = self.internal_loss_weight[i] * bdry_loss_pyramid

                if sm_loss:
                    sm_loss = self.internal_loss_weight[i] * sm_loss_pyramid + sm_loss
                else:
                    sm_loss = self.internal_loss_weight[i] * sm_loss_pyramid
            assert bdry_loss
            assert sm_loss

            smooth_l1 = None
            for i in range(len(predictions)):  # for each pyramid
                if smooth_l1:
                    smooth_l1 = smooth_l1 + self.internal_loss_weight[i] * \
                                (self.hourglass_loss_weight[0] *
                                 F.smooth_l1_loss(predictions[i][0][dis_mask_bool], dis_targets[dis_mask_bool],
                                                  reduction='mean') +
                                 self.hourglass_loss_weight[1] *
                                 F.smooth_l1_loss(predictions[i][1][dis_mask_bool], dis_targets[dis_mask_bool],
                                                  reduction='mean') +
                                 self.hourglass_loss_weight[2] *
                                 F.smooth_l1_loss(predictions[i][2][dis_mask_bool], dis_targets[dis_mask_bool]))
                else:
                    smooth_l1 = self.internal_loss_weight[i] * \
                                (self.hourglass_loss_weight[0] *
                                 F.smooth_l1_loss(predictions[i][0][dis_mask_bool], dis_targets[dis_mask_bool],
                                                  reduction='mean') +
                                 self.hourglass_loss_weight[1] *
                                 F.smooth_l1_loss(predictions[i][1][dis_mask_bool], dis_targets[dis_mask_bool],
                                                  reduction='mean') +
                                 self.hourglass_loss_weight[2] *
                                 F.smooth_l1_loss(predictions[i][2][dis_mask_bool], dis_targets[dis_mask_bool]))
            assert smooth_l1

            loss = self.guided_loss_weight[0] * sm_loss + self.guided_loss_weight[1] * bdry_loss + \
                   self.guided_loss_weight[2] * smooth_l1

        elif self.loss_type == "smoothL1_only":
            smooth_l1 = None
            for i in range(len(predictions)):  # for each pyramid
                if smooth_l1:
                    smooth_l1 = smooth_l1 + self.internal_loss_weight[i] * \
                                (self.hourglass_loss_weight[0] *
                                 F.smooth_l1_loss(predictions[i][0][dis_mask_bool], dis_targets[dis_mask_bool],
                                                  reduction='mean') +
                                 self.hourglass_loss_weight[1] *
                                 F.smooth_l1_loss(predictions[i][1][dis_mask_bool], dis_targets[dis_mask_bool],
                                                  reduction='mean') +
                                 self.hourglass_loss_weight[2] *
                                 F.smooth_l1_loss(predictions[i][2][dis_mask_bool], dis_targets[dis_mask_bool]))
                else:
                    smooth_l1 = self.internal_loss_weight[i] * \
                                (self.hourglass_loss_weight[0] *
                                 F.smooth_l1_loss(predictions[i][0][dis_mask_bool], dis_targets[dis_mask_bool],
                                                  reduction='mean') +
                                 self.hourglass_loss_weight[1] *
                                 F.smooth_l1_loss(predictions[i][1][dis_mask_bool], dis_targets[dis_mask_bool],
                                                  reduction='mean') +
                                 self.hourglass_loss_weight[2] *
                                 F.smooth_l1_loss(predictions[i][2][dis_mask_bool], dis_targets[dis_mask_bool]))
            assert smooth_l1

            loss = smooth_l1

        else:
            raise ValueError("Unexpected loss type: %s" % self.loss_type)

        losses = {"loss_dis": loss * self.loss_weight}
        return losses
