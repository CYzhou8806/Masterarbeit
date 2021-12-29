#!/usr/bin/python
#
# Converts the *instanceIds.png annotations of the Cityscapes dataset
# to COCO-style panoptic segmentation format (http://cocodataset.org/#format-data).
# The convertion is working for 'fine' set of the annotations.
#
# By default with this tool uses IDs specified in labels.py. You can use flag
# --use-train-id to get train ids for categories. 'ignoreInEval' categories are
# removed during the conversion.
#
# In panoptic segmentation format image_id is used to match predictions and ground truth.
# For cityscapes image_id has form <city>_123456_123456 and corresponds to the prefix
# of cityscapes image files.
#

# python imports
from __future__ import print_function, absolute_import, division, unicode_literals
import os
import glob
import random
import shutil
import sys
import argparse
import json
import numpy as np

# Image processing
from PIL import Image

# cityscapes imports
from kitti360scripts.helpers.csHelpers import printError
from kitti360scripts.helpers.labels import Label

#os.environ['KITTI360_DATASET'] = r"D:\Masterarbeit\dataset\kitti_360"
os.environ['KITTI360_DATASET'] = "/bigwork/nhgnycao/datasets/KITTI-360"
#os.environ['KITTI360_DATASET'] = "/media/eistrauben/Dinge/Masterarbeit/dataset/kitti_360"


labels = [
    #       name                     id    kittiId,    trainId   category            catId     hasInstances   ignoreInEval   ignoreInInst   color
    Label(  'unlabeled'            ,  0 ,       -1 ,       255 , 'void'            , 0       , False        , True         , True         , (  0,  0,  0) ),
    Label(  'ego vehicle'          ,  1 ,       -1 ,       255 , 'void'            , 0       , False        , True         , True         , (  0,  0,  0) ),
    Label(  'rectification border' ,  2 ,       -1 ,       255 , 'void'            , 0       , False        , True         , True         , (  0,  0,  0) ),
    Label(  'out of roi'           ,  3 ,       -1 ,       255 , 'void'            , 0       , False        , True         , True         , (  0,  0,  0) ),
    Label(  'static'               ,  4 ,       -1 ,       255 , 'void'            , 0       , False        , False         , True         , (  0,  0,  0) ),
    Label(  'dynamic'              ,  5 ,       -1 ,       255 , 'void'            , 0       , False        , False         , True         , (111, 74,  0) ),
    Label(  'ground'               ,  6 ,       -1 ,       255 , 'void'            , 0       , False        , True         , True         , ( 81,  0, 81) ),
    Label(  'road'                 ,  7 ,        1 ,         0 , 'flat'            , 1       , False        , True        , False        , (128, 64,128) ),
    Label(  'sidewalk'             ,  8 ,        3 ,         1 , 'flat'            , 1       , False        , True        , False        , (244, 35,232) ),
    Label(  'parking'              ,  9 ,        2 ,       255 , 'flat'            , 1       , False        , True         , True         , (250,170,160) ),
    Label(  'rail track'           , 10 ,        10,       255 , 'flat'            , 1       , False        , False         , True         , (230,150,140) ),
    Label(  'building'             , 11 ,        11,         2 , 'construction'    , 2       , True         , False        , False        , ( 70, 70, 70) ),
    Label(  'wall'                 , 12 ,        7 ,         3 , 'construction'    , 2       , False        , False        , False        , (102,102,156) ),
    Label(  'fence'                , 13 ,        8 ,         4 , 'construction'    , 2       , False        , False        , False        , (190,153,153) ),
    Label(  'guard rail'           , 14 ,        30,       255 , 'construction'    , 2       , False        , False         , True         , (180,165,180) ),
    Label(  'bridge'               , 15 ,        31,       255 , 'construction'    , 2       , False        , False         , True         , (150,100,100) ),
    Label(  'tunnel'               , 16 ,        32,       255 , 'construction'    , 2       , False        , False         , True         , (150,120, 90) ),
    Label(  'pole'                 , 17 ,        21,         5 , 'object'          , 3       , True         , False        , True         , (153,153,153) ),
    Label(  'polegroup'            , 18 ,       -1 ,       255 , 'object'          , 3       , False        , False         , True         , (153,153,153) ),
    Label(  'traffic light'        , 19 ,        23,         6 , 'object'          , 3       , True         , False        , True         , (250,170, 30) ),
    Label(  'traffic sign'         , 20 ,        24,         7 , 'object'          , 3       , True         , False        , True         , (220,220,  0) ),
    Label(  'vegetation'           , 21 ,        5 ,         8 , 'nature'          , 4       , False        , True        , False        , (107,142, 35) ),
    Label(  'terrain'              , 22 ,        4 ,         9 , 'nature'          , 4       , False        , True        , False        , (152,251,152) ),
    Label(  'sky'                  , 23 ,        9 ,        10 , 'sky'             , 5       , False        , False        , False        , ( 70,130,180) ),
    Label(  'person'               , 24 ,        19,        11 , 'human'           , 6       , True         , False        , False        , (220, 20, 60) ),
    Label(  'rider'                , 25 ,        20,        12 , 'human'           , 6       , True         , False        , False        , (255,  0,  0) ),
    Label(  'car'                  , 26 ,        13,        13 , 'vehicle'         , 7       , True         , False        , False        , (  0,  0,142) ),
    Label(  'truck'                , 27 ,        14,        14 , 'vehicle'         , 7       , True         , False        , False        , (  0,  0, 70) ),
    Label(  'bus'                  , 28 ,        34,        15 , 'vehicle'         , 7       , True         , False        , False        , (  0, 60,100) ),
    Label(  'caravan'              , 29 ,        16,       255 , 'vehicle'         , 7       , True         , False         , True         , (  0,  0, 90) ),
    Label(  'trailer'              , 30 ,        15,       255 , 'vehicle'         , 7       , True         , False         , True         , (  0,  0,110) ),
    Label(  'train'                , 31 ,        33,        16 , 'vehicle'         , 7       , True         , False        , False        , (  0, 80,100) ),
    Label(  'motorcycle'           , 32 ,        17,        17 , 'vehicle'         , 7       , True         , False        , False        , (  0,  0,230) ),
    Label(  'bicycle'              , 33 ,        18,        18 , 'vehicle'         , 7       , True         , False        , False        , (119, 11, 32) ),

    Label(  'garage'               , 34 ,        12,         2 , 'construction'    , 2       , True         , False         , True         , ( 64,128,128) ),
    Label(  'gate'                 , 35 ,        6 ,         4 , 'construction'    , 2       , False        , False         , True         , (190,153,153) ),
    Label(  'stop'                 , 36 ,        29,       255 , 'construction'    , 2       , True         , False         , True         , (150,120, 90) ),
    Label(  'smallpole'            , 37 ,        22,         5 , 'object'          , 3       , True         , False         , True         , (153,153,153) ),
    Label(  'lamp'                 , 38 ,        25,       255 , 'object'          , 3       , True         , False         , True         , (0,   64, 64) ),
    Label(  'trash bin'            , 39 ,        26,       255 , 'object'          , 3       , True         , False         , True         , (0,  128,192) ),
    Label(  'vending machine'      , 40 ,        27,       255 , 'object'          , 3       , True         , False         , True         , (128, 64,  0) ),
    Label(  'box'                  , 41 ,        28,       255 , 'object'          , 3       , True         , False         , True         , (64,  64,128) ),
    Label(  'unknown construction' , 42 ,        35,       255 , 'void'            , 0       , False        , False         , True         , (102,  0,  0) ),
    Label(  'unknown vehicle'      , 43 ,        36,       255 , 'void'            , 0       , False        , False         , True         , ( 51,  0, 51) ),
    Label(  'unknown object'       , 44 ,        37,       255 , 'void'            , 0       , False        , False         , True         , ( 32, 32, 32) ),

    Label(  'license plate'        , -1 ,        -1,        -1 , 'vehicle'         , 7       , False        , True         , True         , (  0,  0,142) ),
]

id2label        = { label.id      : label for label in labels           }


# The main method
def convert2panGuided(kitti360Path=None, outputRoot=None, useTrainId=False, setNames=None):
    # Where to look for kitti360
    if setNames is None:
        setNames = {"train": [0, 3, 5, 6, 7, 10], "test": [2, 4, 9]}
    if kitti360Path is None:
        if 'KITTI360_DATASET' in os.environ:
            kitti360Path = os.environ['KITTI360_DATASET']
        else:
            kitti360Path = os.path.join(os.path.dirname(os.path.realpath(__file__)),'..','..')
        kitti360Path = os.path.join(kitti360Path, "data_2d_semantics")  # to the path of data_2d_semantics

    if outputRoot is None:
        outputRoot = kitti360Path

    categories = []
    for label in labels:
        if label.ignoreInEval:
            continue
        categories.append({'id': int(label.trainId) if useTrainId else int(label.id),
                           'name': label.name,
                           'color': label.color,
                           'supercategory': label.category,
                           'isthing': 1 if label.hasInstances else 0})

    files = []
    for setName, seqs in setNames.items():
        outputFolder = os.path.join(outputRoot, setName)
        if not os.path.exists(outputFolder):
            os.makedirs(outputFolder)

        for seq in seqs:
            '''
            for root, dir, files in os.walk(os.path.join(kitti360Path, setName)):
                for file in files:
                    if os.path.splitext(file)[-1] == '.png' and root.split('/')[-1] == 'semantic':
            '''

            sequence = '2013_05_28_drive_%04d_sync' % seq

            # how to search for all ground truth
            searchFine = os.path.join(kitti360Path, "*", sequence, "image_00", "instance", "*.png")
            # search files
            filesFine = glob.glob(searchFine)
            filesFine.sort()

            files.extend(filesFine)
            # quit if we did not find anything
            if not filesFine:
                printError(
                    "Did not find any files for {} set using matching pattern {}. Please consult the README.".format(
                        sequence, searchFine)
                )
        # a bit verbose
        print("Converting {} annotation files for {} set.".format(len(files), setName))

        outputBaseFile = "panGuided"
        panGuidedFolder = os.path.join(outputFolder, outputBaseFile)
        if not os.path.exists(panGuidedFolder):
            print("Creating folder {} for {} panoptic segmentation PNGs".format(panGuidedFolder, setName))
            os.makedirs(panGuidedFolder)
        else:
            shutil.rmtree(panGuidedFolder)
            print("---  del old folder...  ---")
            print("Creating folder {} for {} panoptic segmentation PNGs".format(panGuidedFolder, setName))
            os.makedirs(panGuidedFolder)

        print("Corresponding {} segmentations in .png format will be saved in {}".format(setName, panGuidedFolder))

        for progress, f in enumerate(files):  # open the single *instanceIds.png
            originalFormat = np.array(Image.open(f))
            fileName = os.path.basename(f)
            seq = os.path.split(os.path.split(os.path.split(os.path.split(f)[0])[0])[0])[-1]
            outputFileName = seq + '_' + fileName.replace(".png", "_panGuided.png")

            # TODO: change chanels
            pan_format = np.zeros(
                (originalFormat.shape[0], originalFormat.shape[1], 3), dtype=np.uint8
            )
            segmentIds = np.unique(originalFormat)
            num_segmentIds = len(segmentIds)
            grey_value_interval = 256//(num_segmentIds+1)
            for i, segmentId in enumerate(segmentIds):
                if segmentId < 1000:
                    semanticId = segmentId
                else:
                    semanticId = segmentId // 1000
                labelInfo = id2label[semanticId]

                # get all pixel of this semantic class (No distinction between different instances)
                mask = originalFormat == segmentId

                if labelInfo.ignoreInEval:
                    continue
                else:
                    color = grey_value_interval * (i + 1)
                    # pan_format[mask][1] = 1
                    pan_format[mask] = [color, 1, 0]

                # color = [segmentId % 256, segmentId // 256, segmentId // 256 // 256]
                # color = labelInfo.color
                # color = [random.randint(0, 255), random.randint(0,255), random.randint(0,255)]
                # color = grey_value_interval * (i+1)
                # pan_format[mask] = [color, color, color]

            Image.fromarray(pan_format).save(os.path.join(panGuidedFolder, outputFileName))

            print("\rProgress: {:>3.2f} %".format((progress + 1) * 100 / len(files)), end=' ')
            sys.stdout.flush()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-folder",
                        dest="kitti360Path",
                        help="path to the kitti360 dataset 'gtFine' folder",
                        default=None,
                        type=str)
    parser.add_argument("--output-root",
                        dest="outputRoot",
                        help="path to the output folder.",
                        # default=None,
                        default=os.path.join("/bigwork/nhgnycao/Masterarbeit/detectron2/projects/Baseline/datasets", "kitti_360"),
                        # default=os.path.join(os.environ['KITTI360_DATASET'], "kitti_360"),
                        type=str)
    parser.add_argument("--use-train-id", action="store_true", dest="useTrainId", default=False)
    parser.add_argument("--set-names",
                        dest="setNames",
                        help="set names to which apply the function to",
                        nargs='+',
                        default={"train": [0, 3, 5, 6, 7, 10], "test": [2, 4, 9]},
                        # default={"train": [0,], },
                        type=str)
    args = parser.parse_args()

    convert2panGuided(args.kitti360Path, args.outputRoot, args.useTrainId, args.setNames)


# call the main
if __name__ == "__main__":
    main()
