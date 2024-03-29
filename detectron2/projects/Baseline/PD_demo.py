import argparse
import glob
import os
import shutil

from tqdm import tqdm
import time
import cv2
import torch

import os, sys

sys.path.append(os.getcwd())

from train_net import setup
from detectron2.data.detection_utils import read_image
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.utils.logger import setup_logger
from demo.predictor import JointVisualizationDemo
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import detectron2.data.transforms as T
from detectron2.projects.MA import (
    add_joint_estimation_config,
    register_all_cityscapes_joint,
    register_all_sceneflow,
    register_all_sceneflow_flying3d,
    register_all_kitti_2015,
    register_all_kitti360,
    JointDeeplabDatasetMapper,
    JointEvaluator,
)
from evaluation import eval_disparity

# constants
WINDOW_NAME = "demo"


def get_parser_kitti2015():
    parser = argparse.ArgumentParser(description="Detectron2 demo for builtin configs")
    parser.add_argument(
        "--config-file",
        default="./configs/Cityscapes-PanopticSegmentation/demo_joint.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--webcam", action="store_true", help="Take inputs from webcam.")
    parser.add_argument("--video-input", help="Path to video file.")
    parser.add_argument(
        "--input_dir",
        default="datasets/kitti_2015/data_scene_flow/test/image_2",
        help="A list of space separated input images; "
             "or a single glob pattern such as 'directory/*.jpg'",
    )
    # 000004_10_crop
    parser.add_argument(
        "--input_right",
        default=
            "datasets/kitti_2015/data_scene_flow/training/image_3/000020_10.png",
        help="A list of space separated input images; "
             "or a single glob pattern such as 'directory/*.jpg'",
    )
    # default=[r"C:\Users\cyzho\Desktop\data_scene_flow\training\image_2\000004_10.png"],

    parser.add_argument(
        "--output",
        default="data/group_2/evaluations/allAdapt/pan_basePre",
        help="A file or directory to save output visualizations. "
             "If not given, will show output in an OpenCV window.",
    )
    # default=r"C:\Users\cyzho\Desktop\000004_10_seg.png",
    # default="/home/eistrauben/桌面/000004_10_seg.png",
    # default="/bigwork/nhgnycao/Masterarbeit/detectron2/projects/Baseline/demo_output/000004_10_seg.png",
    # /bigwork/nhgnycao/Masterarbeit/detectron2/projects/Baseline/demo_output/aachen_000000_000019_seg.png
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=['MODEL.WEIGHTS', 'data/group_2/weights/allAdapt/pan_basePre/tmp.pth'],
        #default=['MODEL.WEIGHTS', 'model/tmp1.pth'],
        nargs=argparse.REMAINDER,
    )
    return parser


def get_parser_kitti360():
    parser = argparse.ArgumentParser(description="Detectron2 demo for builtin configs")
    parser.add_argument(
        "--config-file",
        default="./configs/Cityscapes-PanopticSegmentation/demo_joint.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--webcam", action="store_true", help="Take inputs from webcam.")
    parser.add_argument("--video-input", help="Path to video file.")
    parser.add_argument(
        "--input_dir",
        default="/home/eistrauben/github/Masterarbeit/detectron2/projects/Baseline/datasets/kitti_360_demo/left",
        help="A list of space separated input images; "
             "or a single glob pattern such as 'directory/*.jpg'",
    )
    # 000004_10_crop
    parser.add_argument(
        "--input_right",
        default=
            "/home/eistrauben/github/Masterarbeit/detectron2/projects/Baseline/datasets/kitti_360_demo/right",
        help="A list of space separated input images; "
             "or a single glob pattern such as 'directory/*.jpg'",
    )
    # default=[r"C:\Users\cyzho\Desktop\data_scene_flow\training\image_2\000004_10.png"],

    parser.add_argument(
        "--output",
        default="output/prediction_kitti360",
        help="A file or directory to save output visualizations. "
             "If not given, will show output in an OpenCV window.",
    )
    # default=r"C:\Users\cyzho\Desktop\000004_10_seg.png",
    # default="/home/eistrauben/桌面/000004_10_seg.png",
    # default="/bigwork/nhgnycao/Masterarbeit/detectron2/projects/Baseline/demo_output/000004_10_seg.png",
    # /bigwork/nhgnycao/Masterarbeit/detectron2/projects/Baseline/demo_output/aachen_000000_000019_seg.png
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=['MODEL.WEIGHTS', 'model/base_kitti2015_360_dis.pth'],
        #default=['MODEL.WEIGHTS', 'model/init_panoptic_cityscapes_weights.pth'],
        nargs=argparse.REMAINDER,
    )
    return parser


def get_parser_diy(input_diy, output_diy, input_right):
    parser = argparse.ArgumentParser(description="Detectron2 demo for builtin configs")
    parser.add_argument(
        "--config-file",
        default="./configs/Cityscapes-PanopticSegmentation/demo_joint.yaml",
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
        "--input_right",
        default=
        input_right,
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
        default=['MODEL.WEIGHTS', 'model/model_0139999.pth'],
        nargs=argparse.REMAINDER,
    )
    return parser


def main_kitti2015(args, eval=False):
    # 设定log文件
    setup_logger(name="fvcore")
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    if os.path.exists(args.output):
        shutil.rmtree(args.output)
    os.makedirs(args.output)

    _root = os.getenv("DETECTRON2_DATASETS", "datasets")
    register_all_cityscapes_joint(_root)
    register_all_sceneflow(_root)
    register_all_kitti_2015(_root)
    register_all_kitti360(_root)
    register_all_sceneflow_flying3d(_root)

    cfg = setup(args)  # 配置设置
    demo = JointVisualizationDemo(cfg)  # $$$ 数据集的处理仍然不清楚

    if args.input_dir:

        for root, dirs, files in os.walk(args.input_dir):
            for file in tqdm(files):
                # if os.path.splitext(file)[0][-1] != '1':
                if True:
                    path = os.path.join(root, file)
                    # use PIL, to be consistent with evaluation
                    img = read_image(path, format="BGR")
                    right_path = path.replace("image_2", "image_3")
                    img_right = read_image(right_path, format="BGR")

                    start_time = time.time()
                    predictions, visualized_output, panoptic_eval = demo.run_on_image(img, img_right, file)
                    logger.info(
                        "{}: {} in {:.2f}s".format(
                            path,
                            "detected {} instances".format(len(predictions["instances"]))
                            if "instances" in predictions
                            else "finished",
                            time.time() - start_time,
                        )
                    )
                    dis_est = predictions['dis_est']
                    dis_est = (dis_est * 256).astype('uint16')
                    dis_est = dis_est.astype('uint16')
                    dis_img = Image.fromarray(dis_est)

                    if args.output:
                        if os.path.isdir(args.output):
                            assert os.path.isdir(args.output), args.output
                            disp_save_dir = os.path.join(args.output, 'dis')
                            if not os.path.exists(disp_save_dir):
                                os.makedirs(disp_save_dir)
                            out_filename = os.path.join(disp_save_dir, os.path.basename(path))
                            out_disp_name = os.path.join(disp_save_dir, os.path.basename(path))

                            panop_save_dir = os.path.join(args.output, 'seg')
                            if not os.path.exists(panop_save_dir):
                                os.makedirs(panop_save_dir)
                            out_panop_name = os.path.join(args.output, 'seg', os.path.basename(path))

                            panoEval_save_dir = os.path.join(args.output, 'cach')
                            if not os.path.exists(panoEval_save_dir):
                                os.makedirs(panoEval_save_dir)
                            out_panoEval_name = os.path.join(args.output, 'cach', os.path.basename(path))

                        else:
                            assert len(args.input) == 1, "Please specify a directory with args.output"
                            out_filename = args.output
                            out_disp_name = out_filename.replace('seg', 'dis')
                        if visualized_output:
                            visualized_output.save(out_panop_name)
                        if panoptic_eval:
                            panoptic_eval.save(out_panoEval_name, format="PNG")
                        dis_img.save(out_disp_name)
                    else:
                        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
                        cv2.imshow(WINDOW_NAME, visualized_output.get_image()[:, :, ::-1])
                        if cv2.waitKey(0) == 27:
                            break  # esc to quit

        if eval:
            #eval_list = ['disp_occ_0', 'flat', 'construction', 'human', 'nature', 'object', 'vehicle', 'textureless', 'occlusions']
            eval_list = ['disp_occ_0',]
            for eval_type in eval_list:
                dis_ground_truth_dir = args.input_dir.replace('image_2', eval_type)
                save_path_dis_eval_result = os.path.join(args.output, eval_type+'_disparity_evaluation.txt')
                eval_disparity(disp_save_dir, dis_ground_truth_dir, save_path_dis_eval_result)

            panop_eval_results, panop_eval_table = demo.evaluate(panoEval_save_dir, )
            dis_eval_result = os.path.join(args.output, 'disp_occ_0_disparity_evaluation.txt')
            with open(dis_eval_result, 'a') as f:
                f.write('\n\n')
                f.write(panop_eval_table)

    '''
    model = Demoer.build_model(cfg)
    DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
        cfg.MODEL.WEIGHTS, resume=args.resume
    )

    res = Demoer.test(cfg, model)  ## $$$
    '''


def main_kitti360(args):
    # 设定log文件
    setup_logger(name="fvcore")
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    if os.path.exists(args.output):
        shutil.rmtree(args.output)
    os.makedirs(args.output)

    dis_path = os.path.join(args.output,"dis")
    seg_path = os.path.join(args.output, "seg")
    os.makedirs(dis_path)
    os.makedirs(seg_path)

    cfg = setup(args)  # 配置设置
    demo = JointVisualizationDemo(cfg)  # $$$ 数据集的处理仍然不清楚

    if args.input_dir:
        for root, dirs, files in os.walk(args.input_dir):
            for file in tqdm(files):
                # if os.path.splitext(file)[0][-1] != '1':
                if True:
                    path = os.path.join(root, file)
                    # use PIL, to be consistent with evaluation
                    img = read_image(path, format="BGR")
                    right_path = path.replace("left", "right")
                    img_right = read_image(right_path, format="BGR")

                    start_time = time.time()
                    predictions, visualized_output = demo.run_on_image(img, img_right)
                    logger.info(
                        "{}: {} in {:.2f}s".format(
                            path,
                            "detected {} instances".format(len(predictions["instances"]))
                            if "instances" in predictions
                            else "finished",
                            time.time() - start_time,
                        )
                    )
                    dis_est = predictions['dis_est']
                    # dis_est = (dis_est * 256).astype('uint16')
                    dis_est = dis_est.astype('uint16')
                    dis_img = Image.fromarray(dis_est)

                    if args.output:
                        if os.path.isdir(args.output):
                            assert os.path.isdir(args.output), args.output
                            out_filename = os.path.join(seg_path, os.path.basename(path).replace("left", "seg"))
                            out_disp_name = os.path.join(dis_path, os.path.basename(path).replace("left", "dis"))
                        else:
                            assert len(args.input) == 1, "Please specify a directory with args.output"
                            out_filename = args.output
                            out_disp_name = out_filename.replace('seg', 'dis')
                        if visualized_output:
                            visualized_output.save(out_filename)
                        dis_img.save(out_disp_name)
                    else:
                        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
                        cv2.imshow(WINDOW_NAME, visualized_output.get_image()[:, :, ::-1])
                        if cv2.waitKey(0) == 27:
                            break  # esc to quit

    '''
    model = Demoer.build_model(cfg)
    DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
        cfg.MODEL.WEIGHTS, resume=args.resume
    )

    res = Demoer.test(cfg, model)  ## $$$
    '''


def demo_kitti2015():
    args = get_parser_kitti2015().parse_args()  # 用于预设/捕获命令行配置
    # args = default_argument_parser().parse_args()  # 用于预设/捕获命令行配置, 和上面自定义的get_parser没啥区别
    main_kitti2015(args, eval=True)


def demo_kitti360():
    args = get_parser_kitti360().parse_args()  # 用于预设/捕获命令行配置
    # args = default_argument_parser().parse_args()  # 用于预设/捕获命令行配置, 和上面自定义的get_parser没啥区别
    main_kitti360(args)



def demo_series_input(source_input_gt_root, output_root):
    if os.path.exists(output_root):
        shutil.rmtree(output_root)
    os.makedirs(output_root)

    # get all inputs from depth result
    for root, dirs, files in os.walk(source_input_gt_root):
        for file in tqdm(files):
            file_path = os.path.join(root, file)
            left_img = file_path.replace("disp_occ_0", "image_2")
            right_img = file_path.replace("disp_occ_0", "image_3")

            #output_diy_name = os.path.splitext(file)[0] + '_seg' + '.png'
            #output_diy = os.path.join(output_root, output_diy_name)
            args = get_parser_diy(left_img, output_root, right_img).parse_args()
            main_kitti2015(args)
            torch.cuda.empty_cache()
    

if __name__ == "__main__":
    demo_kitti2015()
    #demo_kitti360()

    depth_result_root = "datasets/data_scene_flow/kitti_worse_20"
    series_input_gt_root = "datasets/kitti_2015/data_scene_flow/training/disp_occ_0"
    output_dir = "output/predictions"

    # demo_series_input(series_input_gt_root, output_dir)
