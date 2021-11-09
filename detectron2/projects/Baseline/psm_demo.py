import argparse
import glob
import os
import tqdm
import time
import cv2
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.nn.functional as F
from PIL import Image

from train_psm import setup
from detectron2.data.detection_utils import read_image
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.utils.logger import setup_logger
from demo.predictor import VisualizationDemo

# constants
WINDOW_NAME = "demo"


def get_parser():
    parser = argparse.ArgumentParser(description='PSMNet')
    parser.add_argument('--KITTI', default='2015',
                        help='KITTI version')
    parser.add_argument('--datapath', default='/media/jiaren/ImageNet/data_scene_flow_2015/testing/',
                        help='select model')
    parser.add_argument('--loadmodel', default='./weights/pretrained_model_KITTI2015.tar',
                        help='loading model')
    parser.add_argument('--leftimg',
                        default='/home/eistrauben/github/Masterarbeit/datasets/data_scene_flow/training/image_2/000048_10.png',
                        help='load model')
    parser.add_argument('--rightimg',
                        default='/home/eistrauben/github/Masterarbeit/datasets/data_scene_flow/training/image_3/000048_10.png',
                        help='load model')
    parser.add_argument('--model', default='stackhourglass',
                        help='select model')

    parser = argparse.ArgumentParser(description='PSMNet')
    parser.add_argument(
        "--config-file",
        default="./configs/Base-PSM.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--webcam", action="store_true", help="Take inputs from webcam.")
    parser.add_argument("--video-input", help="Path to video file.")
    parser.add_argument('--leftimg',
                        default='/home/eistrauben/github/Masterarbeit/datasets/data_scene_flow/training/image_2/000048_10.png',
                        help='load model')
    parser.add_argument('--rightimg',
                        default='/home/eistrauben/github/Masterarbeit/datasets/data_scene_flow/training/image_3/000048_10.png',
                        help='load model')
    parser.add_argument(
        "--output",
        default="/home/eistrauben/桌面/share/result_panoptic.jpg",
        help="A file or directory to save output visualizations. "
             "If not given, will show output in an OpenCV window.",
    )

    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=['MODEL.WEIGHTS', 'model/model_final_23d03a.pkl'],
        nargs=argparse.REMAINDER,
    )

    parser.add_argument('--maxdisp', type=int, default=192,
                        help='maxium disparity')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')

    return parser


def test(imgL, imgR):
    model.eval()

    if args.cuda:
        imgL = imgL.cuda()
        imgR = imgR.cuda()

    with torch.no_grad():
        disp = model(imgL, imgR)

    disp = torch.squeeze(disp)
    pred_disp = disp.data.cpu().numpy()

    return pred_disp


def main(args):
    # 设定log文件
    setup_logger(name="fvcore")
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    cfg = setup(args)  # 配置设置

    # data preprocess from psm
    normal_mean_var = {'mean': [0.485, 0.456, 0.406],
                       'std': [0.229, 0.224, 0.225]}
    infer_transform = transforms.Compose([transforms.ToTensor(),
                                          transforms.Normalize(**normal_mean_var)])
    imgL_o = Image.open(args.leftimg).convert('RGB')
    imgR_o = Image.open(args.rightimg).convert('RGB')
    imgL = infer_transform(imgL_o)
    imgR = infer_transform(imgR_o)
    # pad to width and hight to 16 times
    if imgL.shape[1] % 16 != 0:
        times = imgL.shape[1] // 16
        top_pad = (times + 1) * 16 - imgL.shape[1]
    else:
        top_pad = 0
    if imgL.shape[2] % 16 != 0:
        times = imgL.shape[2] // 16
        right_pad = (times + 1) * 16 - imgL.shape[2]
    else:
        right_pad = 0
    imgL = F.pad(imgL, (0, right_pad, top_pad, 0)).unsqueeze(0)
    imgR = F.pad(imgR, (0, right_pad, top_pad, 0)).unsqueeze(0)

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


if __name__ == "__main__":
    args = get_parser().parse_args()  # 用于预设/捕获命令行配置
    # args = default_argument_parser().parse_args()  # 用于预设/捕获命令行配置, 和上面自定义的get_parser没啥区别

    main(args)
