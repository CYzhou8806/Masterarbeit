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
from devkit.helpers.csHelpers import printError
from devkit.helpers.labels import Label

labels = [
    #       name                     id    trainId   category            catId     hasInstances   ignoreInEval   color
    Label('unlabeled', 0, 255, 'void', 0, False, True, (0, 0, 0)),
    Label('ego vehicle', 1, 255, 'void', 0, False, True, (0, 0, 0)),
    Label('rectification border', 2, 255, 'void', 0, False, True, (0, 0, 0)),
    Label('out of roi', 3, 255, 'void', 0, False, True, (0, 0, 0)),
    Label('static', 4, 255, 'void', 0, False, False, (0, 0, 0)),
    Label('dynamic', 5, 255, 'void', 0, False, False, (111, 74, 0)),
    Label('ground', 6, 255, 'void', 0, False, True, (81, 0, 81)),
    Label('road', 7, 0, 'flat', 1, False, True, (128, 64, 128)),
    Label('sidewalk', 8, 1, 'flat', 1, False, True, (244, 35, 232)),
    Label('parking', 9, 255, 'flat', 1, False, True, (250, 170, 160)),
    Label('rail track', 10, 255, 'flat', 1, False, False, (230, 150, 140)),
    Label('building', 11, 2, 'construction', 2, False, False, (70, 70, 70)),
    Label('wall', 12, 3, 'construction', 2, False, False, (102, 102, 156)),
    Label('fence', 13, 4, 'construction', 2, False, False, (190, 153, 153)),
    Label('guard rail', 14, 255, 'construction', 2, False, False, (180, 165, 180)),
    Label('bridge', 15, 255, 'construction', 2, False, False, (150, 100, 100)),
    Label('tunnel', 16, 255, 'construction', 2, False, False, (150, 120, 90)),
    Label('pole', 17, 5, 'object', 3, False, False, (153, 153, 153)),
    Label('polegroup', 18, 255, 'object', 3, False, False, (153, 153, 153)),
    Label('traffic light', 19, 6, 'object', 3, False, False, (250, 170, 30)),
    Label('traffic sign', 20, 7, 'object', 3, False, False, (220, 220, 0)),
    Label('vegetation', 21, 8, 'nature', 4, False, True, (107, 142, 35)),
    Label('terrain', 22, 9, 'nature', 4, False, True, (152, 251, 152)),
    Label('sky', 23, 10, 'sky', 5, False, False, (70, 130, 180)),
    Label('person', 24, 11, 'human', 6, True, False, (220, 20, 60)),
    Label('rider', 25, 12, 'human', 6, True, False, (255, 0, 0)),
    Label('car', 26, 13, 'vehicle', 7, True, False, (0, 0, 142)),
    Label('truck', 27, 14, 'vehicle', 7, True, False, (0, 0, 70)),
    Label('bus', 28, 15, 'vehicle', 7, True, False, (0, 60, 100)),
    Label('caravan', 29, 255, 'vehicle', 7, True, False, (0, 0, 90)),
    Label('trailer', 30, 255, 'vehicle', 7, True, False, (0, 0, 110)),
    Label('train', 31, 16, 'vehicle', 7, True, False, (0, 80, 100)),
    Label('motorcycle', 32, 17, 'vehicle', 7, True, False, (0, 0, 230)),
    Label('bicycle', 33, 18, 'vehicle', 7, True, False, (119, 11, 32)),
    Label('license plate', -1, -1, 'vehicle', 7, False, True, (0, 0, 142)),
]

id2label        = { label.id      : label for label in labels           }


# The main method
def convert2panoptic(kitti2015Path=None, outputFolder=None, useTrainId=False, setNames=["val", "train", "test"]):
    # Where to look for Cityscapes

    kitti2015Path = os.path.join(kitti2015Path, "data_scene_flow")

    if outputFolder is None:
        outputFolder = kitti2015Path

    for setName in setNames:  # train, val, test
        # how to search for all ground truth
        searchFine = os.path.join(kitti2015Path, setName, "*", "*_instanceIds.png")
        # search files
        filesFine = glob.glob(searchFine)
        filesFine.sort()

        # get all files that to be converted
        files = filesFine
        # quit if we did not find anything
        if not files:
            printError(
                "Did not find any files for {} set using matching pattern {}. Please consult the README.".format(
                    setName, searchFine)
            )
        # a bit verbose
        print("Converting {} annotation files for {} set.".format(len(files), setName))

        # set the name of the to be generated json
        trainIfSuffix = "_trainId" if useTrainId else ""
        outputBaseFile = "{}{}".format(setName, trainIfSuffix)
        # create the folder for panoptic segmentation PNGs
        panopticFolder = outputFolder
        if not os.path.exists(panopticFolder):
            print("Creating folder {} for {} panoptic segmentation PNGs".format(panopticFolder, setName))
            os.makedirs(panopticFolder)
        else:
            shutil.rmtree(panopticFolder)
            print("---  del old folder...  ---")
            print("Creating folder {} for {} panoptic segmentation PNGs".format(panopticFolder, setName))
            os.makedirs(panopticFolder)

        print("Corresponding segmentations in .png format will be saved in {}".format(panopticFolder))

        for progress, f in enumerate(files):  # open the single *instanceIds.png
            originalFormat = np.array(Image.open(f))
            fileName = os.path.basename(f)
            # panopticFolder = os.path.join(panopticFolder_root, fileName.split('_')[0])
            if not os.path.isdir(panopticFolder):
                os.mkdir(panopticFolder)
            outputFileName = fileName.replace("instanceIds.png", "panGuided.png")

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

            Image.fromarray(pan_format).save(os.path.join(panopticFolder, outputFileName))

            print("\rProgress: {:>3.2f} %".format((progress + 1) * 100 / len(files)), end=' ')
            sys.stdout.flush()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-folder",
                        dest="kitti2015Path",
                        help="path to the Cityscapes dataset 'gtFine' folder",
                        default="/home/eistrauben/github/Masterarbeit/detectron2/projects/Baseline/datasets/kitti_2015",
                        type=str)
    parser.add_argument("--output-folder",
                        dest="outputFolder",
                        help="path to the output folder.",
                        default="/home/eistrauben/github/Masterarbeit/detectron2/projects/Baseline/datasets/kitti_2015/data_scene_flow/val/panGuided",
                        type=str)
    parser.add_argument("--use-train-id", action="store_true", dest="useTrainId")
    parser.add_argument("--set-names",
                        dest="setNames",
                        help="set names to which apply the function to",
                        nargs='+',
                        default=["val",],
                        type=str)
    args = parser.parse_args()

    convert2panoptic(args.kitti2015Path, args.outputFolder, args.useTrainId, args.setNames)


# call the main
if __name__ == "__main__":
    main()
