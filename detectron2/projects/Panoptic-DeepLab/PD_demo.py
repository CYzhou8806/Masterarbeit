import argparse
import glob
import os
import tqdm
import time
import cv2

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
        default="./configs/Cityscapes-PanopticSegmentation/demo_panoptic_deeplab.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--webcam", action="store_true", help="Take inputs from webcam.")
    parser.add_argument("--video-input", help="Path to video file.")
    parser.add_argument(
        "--input",
        nargs="+",
        default="input1.jpg",
        help="A list of space separated input images; "
             "or a single glob pattern such as 'directory/*.jpg'",
    )
    parser.add_argument(
        "--output",
        default="./result1.jpg",
        help="A file or directory to save output visualizations. "
             "If not given, will show output in an OpenCV window.",
    )

    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=["detectron2://model/model_final_23d03a.pkl"],
        nargs=argparse.REMAINDER,
    )
    return parser


def main(args):
    # 设定log文件
    setup_logger(name="fvcore")
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    cfg = setup(args)  # 配置设置
    demo = VisualizationDemo(cfg)   # $$$ 数据集的处理仍然不清楚

    if args.input:
        if len(args.input) == 1:
            args.input = glob.glob(os.path.expanduser(args.input[0]))  # 获取目标文件
            assert args.input, "The input path(s) was not found"
        for path in tqdm.tqdm(args.input, disable=not args.output):
            # use PIL, to be consistent with evaluation
            print(path)
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


if __name__ == "__main__":
    args = get_parser().parse_args()  # 用于预设/捕获命令行配置
    # args = default_argument_parser().parse_args()  # 用于预设/捕获命令行配置, 和上面自定义的get_parser没啥区别

    main(args)
