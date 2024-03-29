#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@Project ：Masterarbeit
@File    ：lab_PD_demo.py
@Author  ：Yu Cao
@Date    ：2021/11/15 13:04 
"""


import argparse
import glob
import os
import shutil

import tqdm
import time
import cv2
import torch

import os, sys

sys.path.append(os.getcwd())

from train_net import setup
from detectron2.data.detection_utils import read_image
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.utils.logger import setup_logger
from demo.predictor import VisualizationDemo

# constants
WINDOW_NAME = "demo"


def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 demo for builtin configs")
    parser.add_argument(
        "--config-file",
        default="./configs/COCO-PanopticSegmentation/panoptic_deeplab_R_52_os16_mg124_poly_200k_bs64_crop_640_640_coco_dsconv.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--webcam", action="store_true", help="Take inputs from webcam.")
    parser.add_argument("--video-input", help="Path to video file.")
    parser.add_argument(
        "--input",
        default=["/bigwork/nhgnycao/lab/datasets/test_lab_01.jpg"],
        help="A list of space separated input images; "
             "or a single glob pattern such as 'directory/*.jpg'",
    )
    parser.add_argument(
        "--output",
        default="/bigwork/nhgnycao/lab/datasets/test_lab_01_seg.jpg",
        help="A file or directory to save output visualizations. "
             "If not given, will show output in an OpenCV window.",
    )

    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=['MODEL.WEIGHTS', 'model/model_final_5e6da2.pkl'],
        nargs=argparse.REMAINDER,
    )
    return parser


def get_parser_diy(input_diy, output_diy):
    parser = argparse.ArgumentParser(description="Detectron2 demo for builtin configs")
    parser.add_argument(
        "--config-file",
        default="./configs/Cityscapes-PanopticSegmentation/demo_panoptic_deeplab.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--webcam", action="store_true", help="Take inputs from webcam.")
    parser.add_argument("--video-input", help="Path to video file.")
    parser.add_argument(
        "--input",
        default=[input_diy],
        help="A list of space separated input images; "
             "or a single glob pattern such as 'directory/*.jpg'",
    )

    parser.add_argument(
        "--output",
        default=output_diy,
        help="A file or directory to save output visualizations. "
             "If not given, will show output in an OpenCV window.",
    )

    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=['MODEL.WEIGHTS', 'model/model_final_23d03a.pkl'],
        nargs=argparse.REMAINDER,
    )
    return parser


def main(args):
    # 设定log文件
    setup_logger(name="fvcore")
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    cfg = setup(args)  # 配置设置
    demo = VisualizationDemo(cfg)  # $$$ 数据集的处理仍然不清楚

    if args.input:
        if len(args.input) == 1:
            args.input = glob.glob(os.path.expanduser(args.input[0]))  # 获取目标文件
            assert args.input, "The input path(s) was not found"
        for path in tqdm.tqdm(args.input, disable=not args.output):
            # use PIL, to be consistent with evaluation
            img = read_image(path, format="BGR")
            start_time = time.time()
            predictions, visualized_output = demo.run_on_image(img)
            logger.info(
                "{}: {} in {:.2f}s".format(
                    path,
                    "detected {} instances".format(len(predictions["instances"]))
                    if "instances" in predictions
                    else "finished",
                    time.time() - start_time,
                )
            )

            if args.output:
                if os.path.isdir(args.output):
                    assert os.path.isdir(args.output), args.output
                    out_filename = os.path.join(args.output, os.path.basename(path))
                else:
                    assert len(args.input) == 1, "Please specify a directory with args.output"
                    out_filename = args.output
                visualized_output.save(out_filename)
            else:
                cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
                cv2.imshow(WINDOW_NAME, visualized_output.get_image()[:, :, ::-1])
                if cv2.waitKey(0) == 27:
                    break  # esc to quit

            return predictions, visualized_output

    '''
    model = Demoer.build_model(cfg)
    DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
        cfg.MODEL.WEIGHTS, resume=args.resume
    )

    res = Demoer.test(cfg, model)  ## $$$
    '''


def demo_single_input():
    args = get_parser().parse_args()  # 用于预设/捕获命令行配置
    # args = default_argument_parser().parse_args()  # 用于预设/捕获命令行配置, 和上面自定义的get_parser没啥区别
    main(args)


def demo_series_input(temple_result_root, source_input_root, output_root):
    if os.path.exists(output_root):
        shutil.rmtree(output_root)
    os.makedirs(output_root)

    # get all inputs from depth result
    for root, dirs, files in os.walk(temple_result_root):
        for file in files:
            input_diy = os.path.join(source_input_root, file)
            output_diy_name = os.path.splitext(file)[0] + '_seg' + '.png'
            output_diy = os.path.join(output_root, output_diy_name)
            args = get_parser_diy(input_diy, output_diy).parse_args()
            main(args)
            torch.cuda.empty_cache()


if __name__ == "__main__":
    demo_single_input()

    depth_result_root = "/bigwork/nhgnycao/Masterarbeit/datasets/data_scene_flow/kitti_worse_20"
    series_input_root = "/bigwork/nhgnycao/Masterarbeit/datasets/data_scene_flow/training/image_2"
    output_dir = "/bigwork/nhgnycao/share/kitti2015_worth20_segments"

    # demo_series_input(depth_result_root, series_input_root, output_dir)


