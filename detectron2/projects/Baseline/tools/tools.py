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
import torch
from tqdm import tqdm
from detectron2.modeling.meta_arch.build import build_model
from detectron2.engine import DefaultTrainer, default_argument_parser
from train_net import setup
import pickle
from detectron2.checkpoint import DetectionCheckpointer



def down_samples_dataset(dataset_root, output_root=None, scale=16):
    for root, dirs, files in os.walk(dataset_root):
        for file in tqdm(files):
            save_dir = root.replace("datasets", "dataset")
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            if os.path.splitext(file)[-1] == ".png":
                file_path = os.path.join(root, file)
                img = cv2.imread(file_path)
                img_down = cv2.resize(img, (img.shape[1] // scale, img.shape[0] // scale),
                                      interpolation=cv2.INTER_NEAREST)
                '''
                cv2.imshow('', img_down)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                '''
                save_path = file_path.replace("datasets", "dataset")
                cv2.imwrite(save_path, img_down)
            else:

                file_path = os.path.join(root, file)
                shutil.copy(file_path, os.path.join(save_dir, file))


if __name__ == "__main__":
    # input_root = "/home/eistrauben/github/Masterarbeit/detectron2/projects/Baseline/datasets/cityscapes"
    input_root = "/bigwork/nhgnycao/Masterarbeit/detectron2/projects/Baseline/datasets/cityscapes"
    # down_samples_dataset(input_root, scale=4)

    '''
    # model = torch.load('init.pth')
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)

    cfg = setup(args)
    model = build_model(cfg)
    model.state_dict()
    for name, para in model.state_dict().items():
        break
    model.load_state_dict(torch.load('model/re_init_panoptic_cityscapes_weights.pth'))


    to_init = {}
    for name, para in model.state_dict().items():
        branch = name.split('.')[0]
        if branch == 'sem_seg_head':
            to_init[name.replace('sem_seg_head', 'dis_embed_head')] = para

    # model.load_state_dict(torch.load('model/model_0139999.pth'))
    checkpointer = DetectionCheckpointer(model)
    checkpointer_5999 = "model/model_0139999.pth"
    checkpointer.load(checkpointer_5999)

    model_dict = model.state_dict()
    state_dict = {k: v for k, v in to_init.items() if k in model_dict.keys()}
    model_dict.update(state_dict)
    model.load_state_dict(model_dict)

    torch.save(model.state_dict(), 're_re_init_panoptic_cityscapes_weights.pth')
    '''

    # model = torch.load('init.pth')
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)

    cfg = setup(args)
    model = build_model(cfg)
    for name, para in model.state_dict().items():
        break
    to_compare = model.load_state_dict(torch.load('model/re_init_panoptic_cityscapes_weights.pth'))

    model_dict = model.state_dict()
    state_dict = {k: v for k, v in to_compare.items() if k in model_dict.keys()}
    model_dict.update(state_dict)
    model.load_state_dict(model_dict)

    torch.save(model.state_dict(), 're_re_init_panoptic_cityscapes_weights.pth')


    model_dict = model.state_dict()
    # state_dict = {k: v for k, v in panoptic_model.items() if k in model_dict.keys()}
    model_dict.update(panoptic_model_dict)
    model.load_state_dict(model_dict)




    '''
    path_panoptic_model_dict = "/home/eistrauben/github/Masterarbeit/detectron2/projects/Baseline/model/original_panoptic_dict.pth"
    panoptic_model_dict = torch.load(path_panoptic_model_dict)
    panoptic_dict_name = list(panoptic_model_dict)

    # path_joint_model_dict = "/home/eistrauben/github/Masterarbeit/detectron2/projects/Baseline/model/model_0059999.pth"
    # joint_model_dict = torch.load(path_joint_model_dict)
    joint_dict_name = list(model.state_dict())

    count = 0
    sum_p = []
    for i, p in enumerate(joint_dict_name):
        sum_p.append(p.split('.')[0])
        print(p.split('.')[0])


    print(sum_p.index('backbone'))
    print(sum_p.index('sem_seg_head'))

    print(sum_p.index('ins_embed_head'))
    print(sum_p.index('dis_embed_head'))
    '''


    '''
    model_dict = model.state_dict()
    # state_dict = {k: v for k, v in panoptic_model.items() if k in model_dict.keys()}
    model_dict.update(panoptic_model_dict)
    model.load_state_dict(model_dict)

    # torch.save(model.state_dict(), 're_init_panoptic_cityscapes_weights.pth')
    # torch.save(model, 'init_panoptic_cityscapes.pth')

    # checkpointer = DetectionCheckpointer(model, save_to_disk=True, save_dir="/home/eistrauben/github/Masterarbeit/detectron2/projects/Baseline/model")
    # checkpointer.save('init_panoptic_cityscapes.pkl')
    #checkpointer = DetectionCheckpointer(model)
    # checkpointer_init = "/home/eistrauben/github/Masterarbeit/detectron2/projects/Baseline/model/init_panoptic_cityscapes.pth"
    # checkpointer.load(checkpointer_init)
    # model.load_state_dict(torch.load(checkpointer_init))
    '''




    print("stop")






