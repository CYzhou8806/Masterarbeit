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
import sys
import argparse
import json
import numpy as np

# Image processing
from PIL import Image

# cityscapes imports
from kitti360scripts.helpers.csHelpers import printError
from kitti360scripts.helpers.labels import id2label, labels

os.environ['KITTI360_DATASET'] = r"D:\Masterarbeit\dataset\kitti_360"


# The main method
def convert2panoptic(kitti360Path=None, outputFolder=None, useTrainId=False, setNames=["val", "train", "test"]):
    # Where to look for kitti360
    if kitti360Path is None:
        if 'KITTI360_DATASET' in os.environ:
            kitti360Path = os.environ['KITTI360_DATASET']
        else:
            kitti360Path = os.path.join(os.path.dirname(os.path.realpath(__file__)),'..','..')
        kitti360Path = os.path.join(kitti360Path, "data_2d_semantics")  # to the path of data_2d_semantics

    if outputFolder is None:
        outputFolder = kitti360Path

    categories = []
    for label in labels:
        if label.ignoreInEval:
            continue
        categories.append({'id': int(label.trainId) if useTrainId else int(label.id),
                           'name': label.name,
                           'color': label.color,
                           'supercategory': label.category,
                           'isthing': 1 if label.hasInstances else 0})

    for setName in setNames:
        '''
        for root, dir, files in os.walk(os.path.join(kitti360Path, setName)):
            for file in files:
                if os.path.splitext(file)[-1] == '.png' and root.split('/')[-1] == 'semantic':
        '''

        # how to search for all ground truth
        searchFine   = os.path.join(kitti360Path, setName, "*", "image_00", "instance", "*.png")
        # search files
        filesFine = glob.glob(searchFine)
        filesFine.sort()

        files = filesFine
        # quit if we did not find anything
        if not files:
            printError(
                "Did not find any files for {} set using matching pattern {}. Please consult the README.".format(setName, searchFine)
            )
        # a bit verbose
        print("Converting {} annotation files for {} set.".format(len(files), setName))

        trainIfSuffix = "_trainId" if useTrainId else ""
        outputBaseFile = "kitti360_panoptic_{}{}".format(setName, trainIfSuffix)
        outFile = os.path.join(outputFolder, "{}.json".format(outputBaseFile))
        print("Json file with the annotations in panoptic format will be saved in {}".format(outFile))
        panopticFolder = os.path.join(outputFolder, outputBaseFile)
        if not os.path.isdir(panopticFolder):
            print("Creating folder {} for panoptic segmentation PNGs".format(panopticFolder))
            os.mkdir(panopticFolder)
        print("Corresponding segmentations in .png format will be saved in {}".format(panopticFolder))

        images = []
        annotations = []
        for progress, f in enumerate(files):

            originalFormat = np.array(Image.open(f))
            seq = os.path.split(os.path.split(os.path.split(os.path.split(f)[0])[0])[0])[-1]
            fileName = os.path.basename(f)
            imageId = seq+'_'+fileName.replace(".png", "")

            inputFileName = seq+'_'+fileName
            outputFileName = seq+'_'+fileName.replace(".png", "_panoptic.png")
            # image entry, id for image is its filename without extension
            images.append({"id": imageId,
                           "width": int(originalFormat.shape[1]),
                           "height": int(originalFormat.shape[0]),
                           "file_name": inputFileName})

            pan_format = np.zeros(
                (originalFormat.shape[0], originalFormat.shape[1], 3), dtype=np.uint8
            )

            segmentIds = np.unique(originalFormat)
            segmInfo = []
            for segmentId in segmentIds:
                if segmentId < 1000:
                    semanticId = segmentId
                    isCrowd = 1
                else:
                    semanticId = segmentId // 1000
                    isCrowd = 0
                labelInfo = id2label[semanticId]
                categoryId = labelInfo.trainId if useTrainId else labelInfo.id
                if labelInfo.ignoreInEval:
                    continue
                if not labelInfo.hasInstances:
                    isCrowd = 0

                mask = originalFormat == segmentId
                color = [segmentId % 256, segmentId // 256, segmentId // 256 // 256]
                pan_format[mask] = color

                area = np.sum(mask) # segment area computation

                # bbox computation for a segment
                hor = np.sum(mask, axis=0)
                hor_idx = np.nonzero(hor)[0]
                x = hor_idx[0]
                width = hor_idx[-1] - x + 1
                vert = np.sum(mask, axis=1)
                vert_idx = np.nonzero(vert)[0]
                y = vert_idx[0]
                height = vert_idx[-1] - y + 1
                bbox = [int(x), int(y), int(width), int(height)]

                segmInfo.append({"id": int(segmentId),
                                 "category_id": int(categoryId),
                                 "area": int(area),
                                 "bbox": bbox,
                                 "iscrowd": isCrowd})

            annotations.append({'image_id': imageId,
                                'file_name': outputFileName,
                                "segments_info": segmInfo})

            Image.fromarray(pan_format).save(os.path.join(panopticFolder, outputFileName))

            print("\rProgress: {:>3.2f} %".format((progress + 1) * 100 / len(files)), end=' ')
            sys.stdout.flush()

        print("\nSaving the json file {}".format(outFile))
        d = {'images': images,
             'annotations': annotations,
             'categories': categories}
        with open(outFile, 'w') as f:
            json.dump(d, f, sort_keys=True, indent=4)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-folder",
                        dest="kitti360Path",
                        help="path to the kitti360 dataset 'gtFine' folder",
                        default=None,
                        type=str)
    parser.add_argument("--output-folder",
                        dest="outputFolder",
                        help="path to the output folder.",
                        default=None,
                        type=str)
    parser.add_argument("--use-train-id", action="store_true", dest="useTrainId", default=False)
    parser.add_argument("--set-names",
                        dest="setNames",
                        help="set names to which apply the function to",
                        nargs='+',
                        default=["train"],
                        type=str)
    args = parser.parse_args()

    convert2panoptic(args.kitti360Path, args.outputFolder, args.useTrainId, args.setNames)


# call the main
if __name__ == "__main__":
    main()
