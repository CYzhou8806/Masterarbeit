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
import shutil
import sys
import argparse
import json
import numpy as np

# Image processing
from PIL import Image

# cityscapes imports
from devkit.helpers.csHelpers import printError
from devkit.helpers.labels import id2label, labels

os.environ[
    'KITTI2015_DATASET'] = "/home/eistrauben/github/Masterarbeit/detectron2/projects/Baseline/datasets/kitti_2015"


# The main method
def convert2panopticMask(kitti2015Path=None, outputFolder=None, useTrainId=False, setNames=None, id_list=None, mod='id'):
    # Where to look for Cityscapes
    if setNames is None:
        setNames = ['test', ]
    if id_list is None:
        id_list = [8, 9, 10, ]
    if kitti2015Path is None:
        if 'KITTI2015_DATASET' in os.environ:
            kitti2015Path = os.environ['KITTI2015_DATASET']
        else:
            kitti2015Path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..')
        kitti2015Path = os.path.join(kitti2015Path, "data_scene_flow")

    if outputFolder is None:
        outputFolder = os.path.join(kitti2015Path, "tmp")


    for setName in setNames:
        # outputFolder = os.path.join(kitti2015Path, setName)
        # how to search for all ground truth
        searchFine = os.path.join(kitti2015Path, setName, "*", "*_instanceIds.png")
        # search files
        filesFine = glob.glob(searchFine)
        filesFine.sort()

        files = filesFine
        # quit if we did not find anything
        if not files:
            printError(
                "Did not find any files for {} set using matching pattern {}. Please consult the README.".format(
                    setName, searchFine)
            )
        # a bit verbose
        print("Converting {} annotation files for {} set.".format(len(files), setName))

        outputBaseFile = "new_flat"
        panopticFolder = os.path.join(outputFolder, outputBaseFile)
        if not os.path.exists(panopticFolder):
            print("Creating folder {} for {} panoptic segmentation PNGs".format(panopticFolder, setName))
            os.makedirs(panopticFolder)
        else:
            shutil.rmtree(panopticFolder)
            print("---  del old folder...  ---")
            print("Creating folder {} for {} panoptic segmentation PNGs".format(panopticFolder, setName))
            os.makedirs(panopticFolder)
        print("Corresponding segmentations in .png format will be saved in {}".format(panopticFolder))

        for progress, f in enumerate(files):
            originalFormat = np.array(Image.open(f))

            fileName = os.path.basename(f)
            outputFileName = fileName.replace("_instanceIds.png", ".png")

            pan_format = np.zeros(
                (originalFormat.shape[0], originalFormat.shape[1], 3), dtype=np.uint8
            )

            segmentIds = np.unique(originalFormat)
            for segmentId in segmentIds:
                if segmentId < 1000:
                    semanticId = segmentId
                else:
                    semanticId = segmentId // 1000
                labelInfo = id2label[semanticId]
                if mod == 'catid':
                    curId = labelInfo.categoryId
                elif mod == 'id':
                    curId = labelInfo.id
                else:
                    raise ValueError('unexcepted mod')

                if curId not in id_list:
                    continue

                mask = originalFormat == segmentId
                color = labelInfo.color
                pan_format[mask] = (255,255,255)

            Image.fromarray(pan_format).save(os.path.join(panopticFolder, outputFileName))
            print("\rProgress: {:>3.2f} %".format((progress + 1) * 100 / len(files)), end=' ')
            sys.stdout.flush()



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-folder",
                        dest="kitti2015Path",
                        help="path to the Cityscapes dataset 'gtFine' folder",
                        default=None,
                        type=str)
    parser.add_argument("--output-folder",
                        dest="outputFolder",
                        help="path to the output folder.",
                        default=None,
                        type=str)
    parser.add_argument("--use-train-id", action="store_true", dest="useTrainId")
    parser.add_argument("--set-names",
                        dest="setNames",
                        help="set names to which apply the function to",
                        nargs='+',
                        default=['test', ],
                        type=str)
    args = parser.parse_args()

    convert2panopticMask(args.kitti2015Path, args.outputFolder, args.useTrainId, args.setNames)


# call the main
if __name__ == "__main__":
    main()
