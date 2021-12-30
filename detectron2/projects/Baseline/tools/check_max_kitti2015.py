import os
from PIL import Image
import numpy as np
from tqdm import tqdm

kitti2015_disparity_dir = "/home/eistrauben/github/Masterarbeit/detectron2/projects/Baseline/datasets/kitti_2015/data_scene_flow/training/disp_occ_0"
max = 0.0
for root, dirs, files in os.walk(kitti2015_disparity_dir):
    for file in tqdm(files):
        dis_gt = Image.open(os.path.join(root, file))
        dis_gt = np.array(dis_gt)
        dis_gt = dis_gt.astype(float)
        dis_gt = dis_gt / 256
        max = max if np.max(dis_gt)<max else np.max(dis_gt)

print(max)

