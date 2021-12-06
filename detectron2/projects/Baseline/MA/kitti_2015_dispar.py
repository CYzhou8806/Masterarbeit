# Copyright (c) Facebook, Inc. and its affiliates.
import json
import logging
import os

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.utils.file_io import PathManager

"""
This file contains functions to register the kitti 2015 to the DatasetCatalog.
"""


logger = logging.getLogger(__name__)


def get_kitti_2015_files(gt_dir):
    output = []
    # files = os.listdir(image_dir)
    for root, dirs, files in os.walk(gt_dir):
        for file in files:
            if os.path.splitext(file)[-1] == '.png':
                gt_path = os.path.join(root, file)
                right_img_path = gt_path.replace('disp_occ_0', 'image_3')
                left_img_path = gt_path.replace('disp_occ_0', 'image_2')
                output.append((left_img_path, right_img_path, gt_path))
    return output


def load_kitti_2015(gt_dir):

    files = get_kitti_2015_files(gt_dir)
    ret = []
    for image_file, right_image_file, disparity_file in files:
        ret.append(
            {
                "file_name": image_file,
                "image_id": "_".join(
                    os.path.splitext(os.path.basename(image_file))[0].split("_")[0]
                ),

                "right_file_name": right_image_file,
                "disparity_file_name_kitti_2015": disparity_file,
            }
        )
    assert len(ret), f"No images found in {gt_dir}!"
    assert PathManager.isfile(
        ret[0]["right_file_name"]
    ), "Please place the right images in the folder"  # noqa
    assert PathManager.isfile(
        ret[0]["disparity_file_name_kitti_2015"]
    ), "Please place the disparity groundturth in the folder"  # noqa
    return ret


_RAW_KITTI_2015_SPLITS = {
    "kitti_2015_train": (
        "kitti_2015/data_scene_flow/training/image_2",
        "kitti_2015/data_scene_flow/training/image_3",
        "kitti_2015/data_scene_flow/training/disp_occ_0",
    ),
    # "test": not supported yet
}


def register_all_kitti_2015(root):
    for key, (image_dir, right_img_dir, gt_dir) in _RAW_KITTI_2015_SPLITS.items():
        image_dir = os.path.join(root, image_dir)
        gt_dir = os.path.join(root, gt_dir)
        right_img_dir = os.path.join(root, right_img_dir)

        DatasetCatalog.register(
            key, lambda x=gt_dir: load_kitti_2015(x)
        )
        MetadataCatalog.get(key).set(
            disparity_root=gt_dir,
            image_root=image_dir,
            right_image_root=right_img_dir,
            evaluator_type="cityscapes_panoptic_seg",
        )
