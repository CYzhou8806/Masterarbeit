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

KITTI2015_CATEGORIES = CITYSCAPES_CATEGORIES


def get_kitti_2015_files(image_dir, gt_dir, json_info):
    output = []
    # files = os.listdir(image_dir)
    image_dict = {}
    for root, dirs, files in os.walk(image_dir):
        for file in files:
            if os.path.splitext(file)[-1] == '.png':
                left_img_path = os.path.join(root, file)

                basename = os.path.basename(left_img_path)
                suffix = ".png"
                assert basename.endswith(suffix), basename
                basename = os.path.basename(basename)[: -len(suffix)]
                image_dict[basename] = left_img_path

    for ann in json_info["annotations"]:
        left_img_path = image_dict.get(ann["image_id"], None)
        assert left_img_path is not None, "No image {} found for annotation {}".format(
            ann["image_id"], ann["file_name"]
        )
        label_file = os.path.join(gt_dir, ann["file_name"])
        segments_info = ann["segments_info"]

        right_img_path = left_img_path.replace('image_2', 'image_3')
        gt_path = left_img_path.replace('image_2', 'disp_occ_0')
        output.append((left_img_path, label_file, segments_info, right_img_path, gt_path))

    assert len(output), "No images found in {}".format(image_dir)
    return output


def load_kitti_2015(image_dir, pan_gt_dir, pan_gt_json, meta):

    def _convert_category_id(segment_info, meta):
        if segment_info["category_id"] in meta["thing_dataset_id_to_contiguous_id"]:
            segment_info["category_id"] = meta["thing_dataset_id_to_contiguous_id"][
                segment_info["category_id"]
            ]
        else:
            segment_info["category_id"] = meta["stuff_dataset_id_to_contiguous_id"][
                segment_info["category_id"]
            ]
        return segment_info

    assert os.path.exists(
        pan_gt_json
    ), "Please run `python devikets_semantics/preparation/createPanopticImgs.py` to generate label files."  # noqa
    with open(pan_gt_json) as f:
        json_info = json.load(f)

    files = get_kitti_2015_files(image_dir, pan_gt_dir, json_info)
    ret = []
    for image_file, label_file, segments_info, right_image_file, disparity_file in files:
        segments_info = [_convert_category_id(x, meta) for x in segments_info]
        pan_guided_file = label_file.replace("panoptic", "panGuided")
        ret.append(
            {
                "file_name": image_file,
                "image_id": "_".join(
                    os.path.splitext(os.path.basename(image_file))[0].split("_")[:2]
                ),
                "pan_seg_file_name": label_file,
                "segments_info": segments_info,

                "right_file_name": right_image_file,
                "disparity_file_name_kitti_2015": disparity_file,
                "pan_guided": pan_guided_file,

            }
        )
    assert len(ret), f"No images found in {image_dir}!"
    assert PathManager.isfile(
        ret[0]["pan_seg_file_name"]
    ), "Please generate panoptic annotation with python kitti360scripts/preparation/createPanopticImgs.py"  # noqa
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
        "kitti_2015/data_scene_flow/training/panoptic",
        "kitti_2015/data_scene_flow/training/panoptic.json",
    ),
    "kitti_2015_val": (
        "kitti_2015/data_scene_flow/val/image_2",
        "kitti_2015/data_scene_flow/val/image_3",
        "kitti_2015/data_scene_flow/val/disp_occ_0",
        "kitti_2015/data_scene_flow/val/panoptic",
        "kitti_2015/data_scene_flow/val/panoptic.json",
    ),
    "kitti_2015_test": (
        "kitti_2015/data_scene_flow/test/image_2",
        "kitti_2015/data_scene_flow/test/image_3",
        "kitti_2015/data_scene_flow/test/disp_occ_0",
        "kitti_2015/data_scene_flow/test/panoptic",
        "kitti_2015/data_scene_flow/test/panoptic.json",
    ),
}


def register_all_kitti_2015(root):
    meta = {}
    # The following metadata maps contiguous id from [0, #thing categories +
    # #stuff categories) to their names and colors. We have to replica of the
    # same name and color under "thing_*" and "stuff_*" because the current
    # visualization function in D2 handles thing and class classes differently
    # due to some heuristic used in Panoptic FPN. We keep the same naming to
    # enable reusing existing visualization functions.
    thing_classes = [k["name"] for k in KITTI2015_CATEGORIES]
    thing_colors = [k["color"] for k in KITTI2015_CATEGORIES]
    stuff_classes = [k["name"] for k in KITTI2015_CATEGORIES]
    stuff_colors = [k["color"] for k in KITTI2015_CATEGORIES]

    meta["thing_classes"] = thing_classes
    meta["thing_colors"] = thing_colors
    meta["stuff_classes"] = stuff_classes
    meta["stuff_colors"] = stuff_colors

    # There are three types of ids in cityscapes panoptic segmentation:
    # (1) category id: like semantic segmentation, it is the class id for each
    #   pixel. Since there are some classes not used in evaluation, the category
    #   id is not always contiguous and thus we have two set of category ids:
    #       - original category id: category id in the original dataset, mainly
    #           used for evaluation.
    #       - contiguous category id: [0, #classes), in order to train the classifier
    # (2) instance id: this id is used to differentiate different instances from
    #   the same category. For "stuff" classes, the instance id is always 0; for
    #   "thing" classes, the instance id starts from 1 and 0 is reserved for
    #   ignored instances (e.g. crowd annotation).
    # (3) panoptic id: this is the compact id that encode both category and
    #   instance id by: category_id * 1000 + instance_id.
    thing_dataset_id_to_contiguous_id = {}
    stuff_dataset_id_to_contiguous_id = {}

    for k in KITTI2015_CATEGORIES:
        if k["isthing"] == 1:
            thing_dataset_id_to_contiguous_id[k["id"]] = k["trainId"]
        else:
            stuff_dataset_id_to_contiguous_id[k["id"]] = k["trainId"]

    meta["thing_dataset_id_to_contiguous_id"] = thing_dataset_id_to_contiguous_id
    meta["stuff_dataset_id_to_contiguous_id"] = stuff_dataset_id_to_contiguous_id

    for key, (image_dir, right_img_dir, dis_gt_dir, pan_gt_dir, gt_json) in _RAW_KITTI_2015_SPLITS.items():
        image_dir = os.path.join(root, image_dir)
        pan_gt_dir = os.path.join(root, pan_gt_dir)
        right_img_dir = os.path.join(root, right_img_dir)
        dis_gt_dir = os.path.join(root, dis_gt_dir)
        gt_json = os.path.join(root, gt_json)

        DatasetCatalog.register(
            key, lambda x=image_dir, y=pan_gt_dir, z=gt_json: load_kitti_2015(x, y, z, meta)
        )
        MetadataCatalog.get(key).set(
            panoptic_root=pan_gt_dir,
            disparity_root=dis_gt_dir,
            panoptic_json=gt_json,
            image_root=image_dir,
            right_image_root=right_img_dir,
            evaluator_type="cityscapes_panoptic_seg",

            ignore_label=255,
            label_divisor=1000,
            **meta,
        )
