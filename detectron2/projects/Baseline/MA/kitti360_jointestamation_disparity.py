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

CITYSCAPES_CATEGORIES = [
    {"color": (128, 64, 128), "isthing": 0, "id": 7, "trainId": 0, "name": "road"},
    {"color": (244, 35, 232), "isthing": 0, "id": 8, "trainId": 1, "name": "sidewalk"},
    {"color": (70, 70, 70), "isthing": 0, "id": 11, "trainId": 2, "name": "building"},
    {"color": (102, 102, 156), "isthing": 0, "id": 12, "trainId": 3, "name": "wall"},
    {"color": (190, 153, 153), "isthing": 0, "id": 13, "trainId": 4, "name": "fence"},
    {"color": (153, 153, 153), "isthing": 0, "id": 17, "trainId": 5, "name": "pole"},
    {"color": (250, 170, 30), "isthing": 0, "id": 19, "trainId": 6, "name": "traffic light"},
    {"color": (220, 220, 0), "isthing": 0, "id": 20, "trainId": 7, "name": "traffic sign"},
    {"color": (107, 142, 35), "isthing": 0, "id": 21, "trainId": 8, "name": "vegetation"},
    {"color": (152, 251, 152), "isthing": 0, "id": 22, "trainId": 9, "name": "terrain"},
    {"color": (70, 130, 180), "isthing": 0, "id": 23, "trainId": 10, "name": "sky"},
    {"color": (220, 20, 60), "isthing": 1, "id": 24, "trainId": 11, "name": "person"},
    {"color": (255, 0, 0), "isthing": 1, "id": 25, "trainId": 12, "name": "rider"},
    {"color": (0, 0, 142), "isthing": 1, "id": 26, "trainId": 13, "name": "car"},
    {"color": (0, 0, 70), "isthing": 1, "id": 27, "trainId": 14, "name": "truck"},
    {"color": (0, 60, 100), "isthing": 1, "id": 28, "trainId": 15, "name": "bus"},
    {"color": (0, 80, 100), "isthing": 1, "id": 31, "trainId": 16, "name": "train"},
    {"color": (0, 0, 230), "isthing": 1, "id": 32, "trainId": 17, "name": "motorcycle"},
    {"color": (119, 11, 32), "isthing": 1, "id": 33, "trainId": 18, "name": "bicycle"},

    # dont need to add these, cause they were ignoreInEval=True
    #{"color": (64, 128, 128), "isthing": 1, "id": 34, "trainId": 19, "name": "garage"},
    #{"color": (190, 153, 153), "isthing": 0, "id": 35, "trainId": 20, "name": "gate"},
    #{"color": (153, 153, 153), "isthing": 1, "id": 37, "trainId": 21, "name": "smallpole"},
]

KITTI360_CATEGORIES = CITYSCAPES_CATEGORIES


def get_kitti360_files(image_dir):
    output = []
    # files = os.listdir(image_dir)
    for root, dirs, files in os.walk(image_dir):
        for file in files:
            if os.path.splitext(file)[-1] == '.png':
                left_img_path = os.path.join(root, file)

                basename = os.path.basename(left_img_path)
                suffix = "_left.png"
                assert basename.endswith(suffix), basename
                basename = os.path.basename(basename)[: -len(suffix)]

                right_img_path = left_img_path.replace('left', 'right')
                gt_path = left_img_path.replace('left', 'disparity')
                gt_path = os.path.splitext(gt_path)[0] + '.tiff'
                output.append([left_img_path, right_img_path, gt_path])
    assert len(output), "No images found in {}".format(image_dir)
    return output


def load_kitti360(image_dir):

    files = get_kitti360_files(image_dir)
    ret = []
    for image_file, right_image_file, disparity_file in files:
        ret.append(
            {
                "file_name": image_file,
                "image_id": "_".join(
                    os.path.splitext(os.path.basename(image_file))[0].split("_")[:-1]
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


_RAW_KITTI360_SPLITS = {
    "kitti360_train": (
        "kitti_360/train/left",
        "kitti_360/train/right",
        "kitti_360/train/disparity",
    ),
}


def register_all_kitti360(root):
    for key, (image_dir, right_img_dir, dis_gt_dir) in _RAW_KITTI360_SPLITS.items():
        image_dir = os.path.join(root, image_dir)
        right_img_dir = os.path.join(root, right_img_dir)
        dis_gt_dir = os.path.join(root, dis_gt_dir)

        DatasetCatalog.register(
            key, lambda x=image_dir: load_kitti360(x)
        )
        MetadataCatalog.get(key).set(
            disparity_root=dis_gt_dir,
            image_root=image_dir,
            right_image_root=right_img_dir,
            evaluator_type="cityscapes_panoptic_seg",
        )


