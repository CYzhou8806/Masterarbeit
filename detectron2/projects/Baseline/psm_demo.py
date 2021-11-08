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
