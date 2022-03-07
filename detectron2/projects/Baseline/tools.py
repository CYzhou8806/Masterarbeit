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
import numpy as np
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

    # model = torch.load('init.pth')
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)

    cfg = setup(args)
    model = build_model(cfg)
    checkpointer = DetectionCheckpointer(model)

    model_dict = model.state_dict()
    to_compare_init = {}
    for name, para in model_dict.items():
        if name.split('.')[0] == 'dis_embed_head' and (name.split('.')[2] == '1/8' or name.split('.')[2] == '1/4') and name.split('.')[3] != 'fusion_block':
            to_compare_init[name] = para.clone()

    to_compare_weight = "model/train_nof_fine_with_init84.pth"
    checkpointer.load(to_compare_weight)
    model_dict = model.state_dict()
    to_compare_weight = {}
    for name, para in model_dict.items():
        if name.split('.')[0] == 'dis_embed_head' and (name.split('.')[2] == '1/8' or name.split('.')[2] == '1/4') and name.split('.')[3] != 'fusion_block':
            to_compare_weight[name] = para.clone()
    count = 0
    for i, [name, para] in enumerate(to_compare_init.items()):
        if (para.equal(to_compare_weight[name])):
            count += 1
            print(name)
    print(count / (i + 1))

    # model_dict.update(to_compare_init)
    # model.load_state_dict(model_dict)
    # torch.save(model.state_dict(), 'model/train_nof_fine_with_init84.pth')



    '''
    # compare the parameter after train step
    checkpointer = DetectionCheckpointer(model)
    checkpointer_16 = "model/train_nof_fine.pth"
    checkpointer.load(checkpointer_16)
    model_dict = model.state_dict()
    to_compare_init = {}
    for name, para in model_dict.items():
        if name.split('.')[0] == 'dis_embed_head' and name.split('.')[2] == '1/8' and name.split('.')[3] != 'fusion_block':
            to_compare_init[name] = para.clone()

    to_compare_weight = "model/split_freez/train_stage4.pth"
    checkpointer.load(to_compare_weight)
    model_dict = model.state_dict()
    to_compare_weight = {}
    for name, para in model_dict.items():
        if name.split('.')[0] == 'dis_embed_head' and name.split('.')[2] == '1/8' and name.split('.')[3] != 'fusion_block':
            # print(para)
            to_compare_weight[name] = para.clone()
    count = 0
    for i, [name, para] in enumerate(to_compare_init.items()):
        if (para.equal(to_compare_weight[name])):
            count +=1
            print(name)
    print(count/(i+1))
    '''

    '''
    tmp = torch.load('model/model_preTrain.pth')['model']
    to_init = {}
    for name, para in tmp.items():
        if name.split('.')[0] == 'dis_embed_head':
            to_init[name] = para
    '''

    # state_dict = {k: v for k, v in tmp.items() if k in model_dict.keys()}
    # model_dict.update(to_init)
    # model.load_state_dict(model_dict)
    # torch.save(model.state_dict(), 'model/old/init.pth')

    '''

    checkpointer = DetectionCheckpointer(model)
    checkpointer_5999 = "output/model_best1.pth"
    checkpointer.load(checkpointer_5999)

    to_init = {}
    for name, para in model.state_dict().items():
        if len(name.split('.')) > 3 and name.split('.')[3] == 'fusion_block':
            to_init[name] = para

    model1 = build_model(cfg)
    model1.state_dict()
    for name1, para1 in model1.state_dict().items():
        break
    checkpointer1 = DetectionCheckpointer(model1)
    checkpointer_59991 = "output/model_best.pth"
    checkpointer1.load(checkpointer_59991)
    model1_dict = model1.state_dict()
    to_init1 = {}
    for name, para in model1_dict.items():
        if len(name.split('.')) > 3 and name.split('.')[3] == 'fusion_block':
            to_init1[name] = para

    count = 0
    for k in to_init1:
        if not to_init1[k].equal(to_init[k]):
            count += 1
            print(k)
    print(count/len(to_init1))
    # state_dict = {k: v for k, v in to_init.items() if k in model1_dict.keys()}
    #model1_dict.update(state_dict)
    #model1.load_state_dict(model1_dict)

    # torch.save(model1.state_dict(), 'model_kitti2015_init.pth')
    '''

    '''
    # model = torch.load('init.pth')
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)

    cfg = setup(args)
    model = build_model(cfg)
    model.state_dict()
    for name, para1 in model.state_dict().items():
        break
    model.load_state_dict(torch.load('model/re_init_panoptic_cityscapes_weights.pth'))

    to_compare_tmp = {}
    to_compare = {}
    for name, para in model.state_dict().items():
        branch = name.split('.')[0]
        if branch == 'dis_embed_head' and name.split('.')[2] == '1/4':
            to_compare[name] = para.clone()
        if branch == 'dis_embed_head' and name.split('.')[2] == '1/16':
            to_compare_tmp[name] = para.clone()

    checkpointer = DetectionCheckpointer(model)
    checkpointer_5999 = "model/model_0004999.pth"
    checkpointer.load(checkpointer_5999)


    model_dict = model.state_dict()
    state_dict = {k: v for k, v in to_compare.items() if k in model_dict.keys()}
    model_dict.update(state_dict)
    model.load_state_dict(model_dict)
    
    # torch.save(model.state_dict(), 're_re_init_panoptic_cityscapes_weights.pth')
    '''

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
