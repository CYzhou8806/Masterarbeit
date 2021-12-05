#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@Project ：Masterarbeit
@File    ：joint_dataset_mapper.py
@Author  ：Yu Cao
@Date    ：2021/11/25 10:05 
"""

import copy
import logging
import numpy as np
from typing import Callable, List, Union, Optional
import torch
from panopticapi.utils import rgb2id
from fvcore.transforms.transform import Transform, TransformList
from detectron2.config import configurable
from detectron2.data import MetadataCatalog
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.data.transforms.augmentation import _transform_to_aug

from .target_generator import PanopticDeepLabTargetGenerator

__all__ = ["JointDeeplabDatasetMapper"]


class AugInputJointEstimation(T.AugInput):
    def __init__(
            self,
            image: np.ndarray,
            *,
            boxes: Optional[np.ndarray] = None,
            sem_seg: Optional[np.ndarray] = None,
            right_img: Optional[np.ndarray] = None,
            dis_gt: Optional[np.ndarray] = None,
            dis_mask: Optional[np.ndarray] = None,
            pan_guid: Optional[np.ndarray] = None,
            pan_mask: Optional[np.ndarray] = None,
    ):
        super().__init__(
            image=image,
            boxes=boxes,
            sem_seg=sem_seg,
        )

        self.right_img = right_img
        self.dis_gt = dis_gt
        self.dis_mask = dis_mask
        self.pan_guid = pan_guid
        self.pan_mask = pan_mask

    def transform(self, tfm: Transform) -> None:
        """
        In-place transform all attributes of this class.

        By "in-place", it means after calling this method, accessing an attribute such
        as ``self.image`` will return transformed data.
        """
        self.image = tfm.apply_image(self.image)
        if self.boxes is not None:
            self.boxes = tfm.apply_box(self.boxes)
        if self.sem_seg is not None:
            self.sem_seg = tfm.apply_segmentation(self.sem_seg)

        if self.right_img is not None:
            self.right_img = tfm.apply_image(self.right_img)
        if self.dis_gt is not None:
            self.dis_gt = tfm.apply_segmentation(self.dis_gt)
        if self.dis_mask is not None:
            self.dis_mask = tfm.apply_segmentation(self.dis_mask)
        if self.pan_guid is not None:
            self.pan_guid = tfm.apply_segmentation(self.pan_guid)
        if self.pan_mask is not None:
            self.pan_mask = tfm.apply_segmentation(self.pan_mask)


class JointDeeplabDatasetMapper:
    """
    The callable currently does the following:

    1. Read the image from "file_name" and label from "pan_seg_file_name"
    2. Applies random scale, crop and flip transforms to image and label
    3. Prepare data to Tensor and generate training targets from label
    """

    @configurable
    def __init__(
            self,
            *,
            augmentations: List[Union[T.Augmentation, T.Transform]],
            image_format: str,
            panoptic_target_generator: Callable,
    ):
        """
        NOTE: this interface is experimental.

        Args:
            augmentations: a list of augmentations or deterministic transforms to apply
            image_format: an image format supported by :func:`detection_utils.read_image`.
            panoptic_target_generator: a callable that takes "panoptic_seg" and
                "segments_info" to generate training targets for the model.
        """
        # fmt: off
        self.augmentations = T.AugmentationList(augmentations)
        self.image_format = image_format
        # fmt: on
        logger = logging.getLogger(__name__)
        logger.info("Augmentations used in training: " + str(augmentations))

        self.panoptic_target_generator = panoptic_target_generator
        self.disparity_target_generator = disparity_target_generator
        self.pan_guided_target_generator = pan_guided_target_generator

    @classmethod
    def from_config(cls, cfg):
        augs = [
            T.ResizeShortestEdge(
                cfg.INPUT.MIN_SIZE_TRAIN,
                cfg.INPUT.MAX_SIZE_TRAIN,
                cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING,
            )
        ]
        if cfg.INPUT.CROP.ENABLED:
            augs.append(T.RandomCrop(cfg.INPUT.CROP.TYPE, cfg.INPUT.CROP.SIZE))
        augs.append(T.RandomFlip())

        # Assume always applies to the training set.
        dataset_names = cfg.DATASETS.TRAIN
        meta = MetadataCatalog.get(dataset_names[0])
        panoptic_target_generator = PanopticDeepLabTargetGenerator(
            ignore_label=meta.ignore_label,
            thing_ids=list(meta.thing_dataset_id_to_contiguous_id.values()),
            sigma=cfg.INPUT.GAUSSIAN_SIGMA,
            ignore_stuff_in_offset=cfg.INPUT.IGNORE_STUFF_IN_OFFSET,
            small_instance_area=cfg.INPUT.SMALL_INSTANCE_AREA,
            small_instance_weight=cfg.INPUT.SMALL_INSTANCE_WEIGHT,
            ignore_crowd_in_semantic=cfg.INPUT.IGNORE_CROWD_IN_SEMANTIC,
        )

        ret = {
            "augmentations": augs,
            "image_format": cfg.INPUT.FORMAT,
            "panoptic_target_generator": panoptic_target_generator,
        }
        return ret

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        # Load image.
        image = utils.read_image(dataset_dict["file_name"], format=self.image_format)
        utils.check_image_size(dataset_dict, image)
        # Panoptic label is encoded in RGB image.
        pan_seg_gt = utils.read_image(dataset_dict.pop("pan_seg_file_name"), "RGB")
        right_image = utils.read_image(dataset_dict["right_file_name"], format=self.image_format)
        dis_gt = utils.read_image(dataset_dict.pop("disparity_file_name"), "RGB")[:, :, 0]
        pan_guided_raw = utils.read_image(dataset_dict.pop("pan_guided"), "RGB")

        aug_input = AugInputJointEstimation(image, right_img=right_image, sem_seg=pan_seg_gt,
                                            dis_gt=dis_gt, pan_guid=pan_guided_raw)
        _ = self.augmentations(aug_input)
        image, pan_seg_gt = aug_input.image, aug_input.sem_seg
        right_image, dis_gt = aug_input.right_img, aug_input.dis_gt
        pan_guided_raw = aug_input.pan_guid, aug_input.pan_mask


        dis_gt = utils.read_image(dataset_dict.pop("disparity_file_name"), "RGB")[:, :, 0]
        dis_gt_with_mask = np.zeros((2, dis_gt.shape[0], dis_gt.shape[1]), dtype=np.float)
        dis_gt = dis_gt.astype(float)
        mask = dis_gt > 0.0
        dis_gt[mask] = (dis_gt[mask] - 1.) / 256
        dis_gt_with_mask[0, :, :] = dis_gt
        dis_gt_with_mask[1][mask] = 1
        valid_dis = dis_gt_with_mask[1, :, :]  # get mask
        valid_dis_mask = valid_dis == 1.0
        mask_max_disp = dis_gt_with_mask[0, :, :] < 192
        mask_disp = np.logical_and(valid_dis_mask, mask_max_disp)

        pan_guided_raw = utils.read_image(dataset_dict.pop("pan_guided"), "RGB")[:, :, :2]
        pan_guided = np.zeros((2, pan_guided_raw.shape[0], pan_guided_raw.shape[1]), dtype=np.float)
        pan_guided[0, :, :] = pan_guided_raw[:, :, 0]
        pan_guided[1, :, :] = pan_guided_raw[:, :, 1]
        pan_mask = pan_guided[1, :, :] == 1.0
        assert pan_guided.shape[0] == 2

        # Reuses crop and transform for dataset.
        # aug_input = T.AugInput(image, sem_seg=pan_seg_gt)
        aug_input = AugInputJointEstimation(image, right_img=right_image, sem_seg=pan_seg_gt,
                                            dis_gt=dis_gt_with_mask[0], dis_mask=mask_disp, pan_guid=pan_guided[0],
                                            pan_mask=pan_mask)
        _ = self.augmentations(aug_input)
        image, pan_seg_gt = aug_input.image, aug_input.sem_seg
        right_img, dis_gt, dis_mask = aug_input.right_img, aug_input.dis_gt, aug_input.dis_mask
        pan_guid, pan_mask = aug_input.pan_guid, aug_input.pan_mask

        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
        dataset_dict["right_image"] = torch.as_tensor(np.ascontiguousarray(right_img.transpose(2, 0, 1)))

        # Generates training targets for Panoptic-DeepLab.
        targets = self.panoptic_target_generator(rgb2id(pan_seg_gt), dataset_dict["segments_info"])
        dataset_dict.update(targets)

        # Generates training targets for disparity.
        dis_target = self.disparity_target_generator(dis_gt, dis_mask)
        dataset_dict.update(dis_target)

        pan_guided_target = self.pan_guided_target_generator(pan_guid, pan_mask)
        dataset_dict.update(pan_guided_target)

        return dataset_dict


'''
def disparity_target_generator(disparity_gt):
    """
     Generates training targets for disparity.
     """
    # TODO: add operations
    return dict(dis_est=torch.as_tensor(np.ascontiguousarray(disparity_gt, dtype=np.float32)),
                )


def pan_guided_target_generator(pan_guided):
    """
     Generates training targets for disparity.
     """
    # TODO: add operations
    return dict(pan_gui=torch.as_tensor(np.ascontiguousarray(pan_guided, dtype=np.float32)),
                )
'''


def disparity_target_generator(disparity_gt, mask):
    """
     Generates training targets for disparity.
     """
    # TODO: add operations
    return dict(dis_est=torch.as_tensor(np.ascontiguousarray(disparity_gt, dtype=np.float32)),
                dis_mask=torch.as_tensor(np.ascontiguousarray(mask, dtype=np.float32)),
                )


def pan_guided_target_generator(pan_guided, mask):
    """
     Generates training targets for disparity.
     """
    # TODO: add operations
    return dict(pan_gui=torch.as_tensor(np.ascontiguousarray(pan_guided, dtype=np.float32)),
                pan_mask=torch.as_tensor(np.ascontiguousarray(mask, dtype=np.float32)),
                )
