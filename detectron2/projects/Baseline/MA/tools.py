#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@Project ：Masterarbeit
@File    ：tools.py
@Author  ：Yu Cao
@Date    ：2021/12/1 19:19 
"""
import os
import shutil

import cv2
from tqdm import tqdm


def down_samples_dataset(dataset_root, output_root=None, scale=16):
    for root, dirs, files in os.walk(dataset_root):
        for file in tqdm(files):
            if os.path.splitext(file)[-1] == ".png":
                file_path = os.path.join(root, file)
                img = cv2.imread(file_path)
                img_down = cv2.resize(img, (img.shape[1]//scale, img.shape[0]//scale), interpolation=cv2.INTER_NEAREST)
                '''
                cv2.imshow('', img_down)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                '''
                save_path = file_path.replace("datasets", "dataset")
                save_dir = root.replace("datasets", "dataset")
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                cv2.imwrite(save_path, img_down)
            else:
                file_path = os.path.join(root, file)
                shutil.copytree(file_path, file_path.replace("datasets", "dataset"))


if __name__ == "__main__":
    # input_root = "/home/eistrauben/github/Masterarbeit/detectron2/projects/Baseline/datasets/cityscapes"
    input_root = "/bigwork/nhgnycao/Masterarbeit/detectron2/projects/Baseline/datasets/cityscapes"
    down_samples_dataset(input_root, scale=16)
