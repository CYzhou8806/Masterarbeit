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
from PIL import Image, ImageFilter
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
            isVal=False,
            *,
            augmentations: List[Union[T.Augmentation, T.Transform]],
            image_format: str,
            panoptic_target_generator: Callable,
            do_aug: bool,
            do_crop: bool,
            panoptic_branch: bool,
            guided_loss: bool,
            max_disp: int,
            augmentations_val: List[Union[T.Augmentation, T.Transform]],
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
        self.augmentations = T.AugmentationList(augmentations if not isVal else augmentations_val)
        self.image_format = image_format
        # fmt: on
        logger = logging.getLogger(__name__)
        logger.info("Augmentations used in training: " + str(augmentations))

        self.panoptic_target_generator = panoptic_target_generator
        self.disparity_target_generator = disparity_target_generator
        self.pan_guided_target_generator = pan_guided_target_generator
        self.do_aug = do_aug
        self.do_crop = do_crop
        self.panoptic_branch = panoptic_branch
        self.guided_loss = guided_loss
        self.max_disp = max_disp

    @classmethod
    def from_config(cls, cfg):
        if cfg.INPUT.DO_AUGUMENTATION and cfg.INPUT.CROP.ENABLED:
            augs = [T.ResizeShortestEdge(
                cfg.INPUT.MIN_SIZE_TRAIN,
                cfg.INPUT.MAX_SIZE_TRAIN,
                cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING,
            ), T.RandomCrop(cfg.INPUT.CROP.TYPE, cfg.INPUT.CROP.SIZE), T.RandomFlip()]
        elif cfg.INPUT.CROP.ENABLED:
            augs = [T.RandomCrop(cfg.INPUT.CROP.TYPE, cfg.INPUT.CROP.SIZE)]
        else:
            augs = [T.ResizeShortestEdge(
                cfg.INPUT.MIN_SIZE_TRAIN,
                cfg.INPUT.MAX_SIZE_TRAIN,
                cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING,
            ), T.RandomFlip()]
        augs_val = [T.RandomCrop(cfg.INPUT.CROP.TYPE, cfg.INPUT.CROP.VAL_SIZE)]
        #augs_val = [T.FixedSizeCrop(cfg.INPUT.CROP.VAL_SIZE, pad=True, pad_value=0.0)]

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
        ) if cfg.MODEL.MODE.PANOPTIC_BRANCH else None

        ret = {
            "augmentations": augs,
            "augmentations_val": augs_val,
            "image_format": cfg.INPUT.FORMAT,
            "panoptic_target_generator": panoptic_target_generator,
            "do_aug": cfg.INPUT.DO_AUGUMENTATION,
            "do_crop": cfg.INPUT.CROP.ENABLED,
            "panoptic_branch": cfg.MODEL.MODE.PANOPTIC_BRANCH,
            "guided_loss": cfg.MODEL.DIS_EMBED_HEAD.LOSS_TYPE == "panoptic_guided",
            "max_disp": cfg.MODEL.DIS_EMBED_HEAD.MAX_DISP,
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
        pan_seg_gt = utils.read_image(dataset_dict.pop("pan_seg_file_name"), "RGB") if self.panoptic_branch else None
        right_image = utils.read_image(dataset_dict["right_file_name"], format=self.image_format)

        cityscapes = False
        kitti_2015 = False
        scene_flow = False
        if "disparity_file_name_tiff" in dataset_dict:
            scene_flow = True
            dis_gt = Image.open(dataset_dict.pop("disparity_file_name_tiff"))
            dis_gt = np.array(dis_gt)
        elif "disparity_file_name" in dataset_dict:
            cityscapes = True
            dis_gt = utils.read_image(dataset_dict.pop("disparity_file_name"), "RGB")[:, :, 0]
        elif "disparity_file_name_kitti_2015" in dataset_dict:
            kitti_2015 = True
            dis_gt = Image.open(dataset_dict.pop("disparity_file_name_kitti_2015"))
            dis_gt = np.array(dis_gt)
            # mask_tmp = dis_gt !=0
            # print(np.sum(dis_gt !=0.0))

            '''
            dis_tmp = np.zeros((dis_gt.shape[0], dis_gt.shape[1],3))
            for i in range(3):
                dis_tmp[:, :, i] = dis_gt

            def numpy2pil(np_array: np.ndarray) -> Image:
                """
                 Convert an HxWx3 numpy array into an RGB Image
                """

                assert_msg = 'Input shall be a HxWx3 ndarray'
                assert isinstance(np_array, np.ndarray), assert_msg
                assert len(np_array.shape) == 3, assert_msg
                assert np_array.shape[2] == 3, assert_msg

                img = Image.fromarray(np_array, 'RGB')
                return img

            dis_tmp = numpy2pil(dis_tmp)
            dis_gt = dis_tmp.filter(ImageFilter.GaussianBlur(radius=2))
            dis_gt = np.array(dis_gt)[:,:,0]
            '''

        else:
            raise TypeError("unexcepted form of disparity ground truth.")

        pan_guided_raw = utils.read_image(dataset_dict.pop("pan_guided"), "RGB") if self.guided_loss else None

        if self.do_aug or self.do_crop:
            # Reuses crop and transform for dataset.
            aug_input = AugInputJointEstimation(image, right_img=right_image, sem_seg=pan_seg_gt,
                                                dis_gt=dis_gt.astype(float), pan_guid=pan_guided_raw)
            _ = self.augmentations(aug_input)
            image, pan_seg_gt = aug_input.image, aug_input.sem_seg
            right_image, dis_gt = aug_input.right_img, aug_input.dis_gt
            pan_guided_raw = aug_input.pan_guid

        dis_gt_with_mask = np.zeros((2, dis_gt.shape[0], dis_gt.shape[1]), dtype=np.float)
        dis_gt = dis_gt.astype(float)
        '''
        mask = dis_gt_with_mask[0] == 0.0   # all position, redundant/useless
        print(np.sum(mask))

        if cityscapes:  # only for cityscapes datasets
            dis_gt[mask] = (dis_gt[mask] - 1.) / 256
        if kitti_2015:  # only for kitti 2015 datasets
            dis_gt[mask] = dis_gt[mask] / 256
        '''
        if cityscapes:  # only for cityscapes datasets
            dis_gt = (dis_gt - 1.) / 256
        if kitti_2015:  # only for kitti 2015 datasets
            dis_gt = dis_gt / 256
        '''
        print(np.sum(dis_gt>0))
        print(np.sum(dis_gt != 0))
        print(np.sum(dis_gt > 0.0))
        print(np.sum(dis_gt != 0.0))
        '''
        dis_gt_with_mask[0, :, :] = dis_gt

        '''
        # TODO:debug_tmp
        neighbor = 1
        size = neighbor * 2
        h, w = dis_gt_with_mask[0].shape
        for i in range(neighbor, h - neighbor, size):
            for j in range(neighbor, w - neighbor, size):
                local = []
                way = []
                for p in range(neighbor + 1):
                    for q in range(p + 1):
                        if p != 0:
                            if [i + p, j + q] not in way:
                                local.append(dis_gt_with_mask[0][i + p, j + q])
                                way.append([i + p, j + q])
                            if [i + p, j - q] not in way:
                                local.append(dis_gt_with_mask[0][i + p, j - q])
                                way.append([i + p, j - q])
                            if [i - p, j + q] not in way:
                                local.append(dis_gt_with_mask[0][i - p, j + q])
                                way.append([i - p, j + q])
                            if [i - p, j - q] not in way:
                                local.append(dis_gt_with_mask[0][i - p, j - q])
                                way.append([i - p, j - q])
                            if [i + q, j + p] not in way:
                                local.append(dis_gt_with_mask[0][i + q, j + p])
                                way.append([i + q, j + p])
                            if [i + q, j - p] not in way:
                                local.append(dis_gt_with_mask[0][i + q, j - p])
                                way.append([i + q, j - p])
                            if [i - q, j + p] not in way:
                                local.append(dis_gt_with_mask[0][i - q, j + p])
                                way.append([i - q, j + p])
                            if [i - q, j - p] not in way:
                                local.append(dis_gt_with_mask[0][i - q, j - p])
                                way.append([i - q, j - p])
                        else:
                            local.append(dis_gt_with_mask[0][i, j])
                            way.append([i, j])
                            break
                tmp = len(local)
                if local.count(0.0) >=1:
                # if 1:
                    local_max = max(local)
                    
                    #for [x, y] in way:
                    #    dis_gt_with_mask[0][x, y] = local_max
                    
                    if dis_gt_with_mask[0][i, j] == 0.0:
                        dis_gt_with_mask[0][i, j] = local_max
        '''
        '''
        dis_gt_with_mask[1] = 1.0
        valid_dis = dis_gt_with_mask[1, :, :]  # get mask
        valid_dis_mask = valid_dis == 1.0
        '''
        mask_max_disp = dis_gt_with_mask[0, :, :] < self.max_disp
        dis_gt_with_mask[1][mask_max_disp] = 1.0
        '''
        print(np.sum(mask_max_disp))
        print(np.sum(dis_gt_with_mask[1]==1.0))
        
        print(np.sum(dis_gt_with_mask[0] > 0.0))
        '''
        mask_disp = mask_max_disp
        # mask_disp = np.logical_and(valid_dis_mask, mask_max_disp)
        # print(np.sum(dis_gt_with_mask[0, :, :]>0))

        if self.guided_loss:
            pan_guided_raw2 = pan_guided_raw[:, :, :2]
            pan_guided = np.zeros((2, pan_guided_raw2.shape[0], pan_guided_raw2.shape[1]), dtype=np.float)
            pan_guided[0, :, :] = pan_guided_raw2[:, :, 0]
            pan_guided[1, :, :] = pan_guided_raw2[:, :, 1]
            pan_mask = pan_guided[1, :, :] == 1.0
            '''
            # TODO:debug
            if not np.any(pan_mask):
                print("no 1")
                Image.fromarray(pan_guided_raw).save("pan_guided_raw.png")
                Image.fromarray(image).save("image.png")
                Image.fromarray(pan_seg_gt).save("pan_seg_gt.png")
                raise RuntimeError("excepted stop")
            '''



            assert pan_guided.shape[0] == 2
            pan_guided_target = self.pan_guided_target_generator(pan_guided[0], pan_mask)
            dataset_dict.update(pan_guided_target)

        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)), dtype=torch.float32)
        dataset_dict["right_image"] = torch.as_tensor(np.ascontiguousarray(right_image.transpose(2, 0, 1)),
                                                      dtype=torch.float32)

        if self.panoptic_branch:
            # Generates training targets for Panoptic-DeepLab.
            targets = self.panoptic_target_generator(rgb2id(pan_seg_gt), dataset_dict["segments_info"])
            dataset_dict.update(targets)

        # Generates training targets for disparity.
        dis_target = self.disparity_target_generator(dis_gt_with_mask[0], mask_disp)
        dataset_dict.update(dis_target)

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
