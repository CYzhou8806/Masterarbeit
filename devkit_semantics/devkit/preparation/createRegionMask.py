import os

import numpy as np
from PIL import Image
import glob
import shutil



def main(gt_dir, region_mask_name, ouput_root):
    searchFine = os.path.join(gt_dir, "*.png")
    filesFine = glob.glob(searchFine)
    filesFine.sort()
    files = filesFine

    save_root = os.path.join(ouput_root, region_mask_name)
    if not os.path.exists(save_root):
        os.makedirs(save_root)
    else:
        shutil.rmtree(save_root)
        print("---  del old folder...  ---")
        os.makedirs(save_root)

    for file in files:
        basename = os.path.basename(file)
        mask_file = Image.open(file.replace('disp_occ_0', region_mask_name))
        region_mask = np.array(mask_file)[:,:,0]
        mask = region_mask == 0
        gt = Image.open(file)
        gt = np.array(gt)

        gt[mask] = 0

        Image.fromarray(gt).save(os.path.join(save_root, basename))






# call the main
if __name__ == "__main__":
    gt_dir = '/home/eistrauben/github/Masterarbeit/detectron2/projects/Baseline/datasets/kitti_2015/data_scene_flow/test/disp_occ_0'
    #region_mask_list = ['flat', 'construction', 'human', 'nature', 'object', 'vehicle']
    ouput_root = '/home/eistrauben/github/Masterarbeit/detectron2/projects/Baseline/datasets/kitti_2015/data_scene_flow/test/mask_gt'
    region_mask_list = ['new_flat', ]
    for region_mask_name in region_mask_list:
        main(gt_dir, region_mask_name, ouput_root)