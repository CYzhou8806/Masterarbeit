# Copyright (c) Facebook, Inc. and its affiliates.
import json
import logging
import os

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.utils.file_io import PathManager

"""
This file contains functions to register the kitti360 driving to the DatasetCatalog.
"""


logger = logging.getLogger(__name__)


def get_kitti360_files(image_dir):
    output = []
    # files = os.listdir(image_dir)
    for root, dirs, files in os.walk(image_dir):
        for file in files:
            if os.path.splitext(file)[-1] == '.png':
                left_img_path = os.path.join(root, file)
                right_img_path = left_img_path.replace('left', 'right')
                gt_path = left_img_path.replace('left', 'disparity').split('.')[0] + '.tiff'
                output.append((left_img_path, right_img_path, gt_path))
    return output


def load_kitti360(image_dir):

    files = get_kitti360_files(image_dir)
    ret = []
    for image_file, right_image_file, disparity_file in files:
        ret.append(
            {
                "file_name": image_file,
                "image_id": "_".join(
                    os.path.splitext(os.path.basename(image_file))[0].split("_")[0]
                ),

                "right_file_name": right_image_file,
                "disparity_file_name_tiff": disparity_file,
            }
        )
    assert len(ret), f"No images found in {image_dir}!"
    assert PathManager.isfile(
        ret[0]["right_file_name"]
    ), "Please place the right images in the folder"  # noqa
    assert PathManager.isfile(
        ret[0]["disparity_file_name_tiff"]
    ), "Please place the disparity groundturth in the folder"  # noqa
    return ret


_RAW_KITTI360_DRIVING_SPLITS = {
    "kitti360_train": (
        "kitti_360/train/left",
        "kitti_360/train/right",
        "kitti_360/train/disparity",
    ),
    # "test": not supported yet
}


def register_all_kitti360(root):
    for key, (image_dir, right_img_dir, gt_dir) in _RAW_KITTI360_DRIVING_SPLITS.items():
        image_dir = os.path.join(root, image_dir)
        gt_dir = os.path.join(root, gt_dir)
        right_img_dir = os.path.join(root, right_img_dir)

        DatasetCatalog.register(
            key, lambda x=image_dir: load_kitti360(x)
        )
        MetadataCatalog.get(key).set(
            disparity_root=gt_dir,
            image_root=image_dir,
            right_image_root=right_img_dir,
            evaluator_type="cityscapes_panoptic_seg",
        )
