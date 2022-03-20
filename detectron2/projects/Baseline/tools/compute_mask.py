from utils.image_io import read, write
import cv2
import numpy as np
from numpy import inf
import os


def disp_remap(disp_gt, negative=True):
    disp_remaped = np.zeros_like(disp_gt)

    shape = disp_gt.shape

    for y in range(shape[0]):
        for x in range(shape[1]):
            disp = disp_gt[y][x]
            x_remap = x - disp if negative else x + disp
            if x_remap >= 0 and x_remap < shape[1]:
                if disp > disp_remaped[y][x_remap]:
                    disp_remaped[y][x_remap] = disp

    return disp_remaped

def compute_occluded_old(disp_gt):
    disp_gt[disp_gt == -inf] = 0
    disp_gt[disp_gt == inf] = 0
    disp_gt = disp_gt.astype(int)
    noc = disp_remap(disp_remap(disp_gt), False)

    #des = cv2.medianBlur(noc.astype(np.float32), 5)

    diff = np.abs(noc - disp_gt)
    mask = np.zeros_like(disp_gt)
    mask[diff > 0] = 255

    kernel = np.ones((3, 3), np.float32)
    des = cv2.filter2D(mask.astype(np.float32), -1, kernel)

    threshold = 6
    mask[des < threshold * 255] = 0
    mask[des >= threshold * 255] = 255
    return mask.astype(np.uint8)

def complex_unique(a):
    weight = 1j*np.linspace(0, a.shape[1], a.shape[0], endpoint=False)
    b = a + weight[:, np.newaxis]
    u, ind = np.unique(b, return_index=True)
    b = np.zeros_like(a)
    np.put(b, ind, a.flat[ind])
    return b

def compute_occluded(disp_gt):
    disp_gt[disp_gt == -inf] = 0
    disp_gt[disp_gt == inf] = 0
    disp_gt = disp_gt.astype(int)

    noc_mask = np.copy(disp_gt)

    shape = disp_gt.shape

    idxs = np.arange(shape[1])
    idxs = np.outer(np.ones(shape[0]), idxs)

    x_remap = idxs - disp_gt
    noc_mask[x_remap < 0] = 0
    noc_mask[x_remap >= shape[1]] = 0
    x_remap_flipped = np.flip(x_remap, axis=1)
    unique = complex_unique(x_remap_flipped)
    unique = np.flip(unique, axis=1)

    noc_mask[unique < 1] = 0

    diff = np.abs(noc_mask - disp_gt)
    mask = np.zeros_like(disp_gt)
    mask[diff > 0] = 255

    kernel = np.ones((3, 3), np.float32)
    des = cv2.filter2D(mask.astype(np.float32), -1, kernel)

    threshold = 6
    mask[des < threshold * 255] = 0
    mask[des >= threshold * 255] = 255
    return mask.astype(np.uint8)


def compute_textureless(left_image, disp_gt, textureless_width=3, textureless_thresh=4):
    """
    :param left_image:
    :param textureless_width: box filter width applied to squared gradient
    :param textureless_thresh: threshold applied to filtered squared gradient
    :return: a numpy array containing the textureless mask
    """
    left_image = left_image.astype(np.float64)
    shape = left_image.shape
    height, width, depth = shape[0], shape[1], shape[2]
    tex_mask = np.zeros((height, width))
    gradients = np.zeros((height, width), dtype=np.float64)

    diff = np.mean(np.square(left_image[:, :-1, :] - left_image[:, 1:, :]), axis=2)
    gradients[:, 1:] = diff
    gradients[:, :-1] = np.maximum(gradients[:, :-1], diff)


    # aggregate with a box filter
    if textureless_width > 0:
        gradients = cv2.boxFilter(gradients, -1, (textureless_width, textureless_width))

    # threshold to get final map
    tex_mask[gradients < textureless_thresh * textureless_thresh] = 255
    # set pixels without gt as black
    tex_mask[disp_gt == 0] = 0
    return tex_mask.astype(np.uint8)


def compute_textureless_old(left_image, disp_gt, textureless_width=3, textureless_thresh=4):
    """
    :param left_image:
    :param textureless_width: box filter width applied to squared gradient
    :param textureless_thresh: threshold applied to filtered squared gradient
    :return: a numpy array containing the textureless mask
    """
    left_image = left_image.astype(np.float64)
    shape = left_image.shape
    height, width, depth = shape[0], shape[1], shape[2]
    tex_mask = np.zeros((height, width))
    gradients = np.zeros((height, width), dtype=np.float64)

    # compute squared horizontal gradient
    for i in range(height):
        for j in range(width-1):
            diff = np.mean(np.square(left_image[i, j, :] - left_image[i, j+1, :]))
            gradients[i, j+1] = diff
            gradients[i, j] = np.maximum(gradients[i, j], diff)

    # aggregate with a box filter
    if textureless_width > 0:
        gradients = cv2.boxFilter(gradients, -1, (textureless_width, textureless_width))

    # threshold to get final map
    tex_mask[gradients < textureless_thresh * textureless_thresh] = 255
    # set pixels without gt as black
    tex_mask[disp_gt == 0] = 0
    return tex_mask.astype(np.uint8)


def compute_disparity_discont(gt_disparity, disp_gap=2.0, discont_width=9):
    """
    compute the mask for disparity discontinuities
    1. threshold horiz. and vert. depth discontinuities with disp_gap
    2. apply a box filter of discont_width
    3. re-threshold above 0
    :param gt_disparity:
    :param disp_gap: disparity jump threshold
    :param discont_width: width of discontinuity region, used as kernel size of box filter
    :return: a numpy array containing the disc mask
    """
    height, width = gt_disparity.shape
    disc_tmp = np.zeros_like(gt_disparity)
    disc_mask = np.zeros_like(gt_disparity)

    # Assure that there are no constructs like -inf, inf
    gt_disparity[gt_disparity == -inf] = 0
    gt_disparity[gt_disparity == inf] = 0

    # find discontinuities
    for i in range(height-1):
        for j in range(width-1):
            if gt_disparity[i, j] and gt_disparity[i, j+1]:
                h_diff = np.abs(int(gt_disparity[i,j]) - int(gt_disparity[i,j+1]))
                if h_diff >= disp_gap:
                    disc_tmp[i,j] = disc_tmp[i,j+1] = 128
            if gt_disparity[i, j] and gt_disparity[i+1, j]:
                v_diff = np.abs(int(gt_disparity[i,j]) - int(gt_disparity[i+1,j]))
                if v_diff >= disp_gap:
                    disc_tmp[i,j] = disc_tmp[i+1,j] = 128

    # aggregate with a box filter
    if discont_width > 0:
        disc_tmp = cv2.boxFilter(disc_tmp, -1, (discont_width, discont_width))

    # Threshold to get final map
    disc_mask[disc_tmp > 0] = 255
    # set pixels without gt as black
    disc_mask[gt_disparity == 0] = 0
    return disc_mask.astype(np.uint8)


def compute_good_region(gt_disparity, mask_tex, mask_disc=None, mask_occ=None):
    """
    set pixels with gt disparity but not in the three special regions to 128
    :param gt_disparity:
    :param mask_tex: mask for textureless region
    :param mask_disc: mask for discontinuities
    :param mask_occ: mask for occlusions
    :return:
    """
    # Assure that there are no constructs like -inf, inf
    gt_disparity[gt_disparity == -inf] = 0
    gt_disparity[gt_disparity == inf] = 0

    mask_good = np.ones_like(gt_disparity)
    mask_good[mask_tex == 128] = 0
    if mask_disc is not None:
        mask_good[mask_disc == 128] = 0
    if mask_occ is not None:
        mask_good[mask_occ == 128] = 0
    mask_good[gt_disparity == 0] = 0
    return mask_good


def recompute_mask_wo_occ(mask, mask_occ):
    """
    :param mask: mask of textureless region or discontinuities
    :param mask_occ: mask for occlusions
    :return: a mask (textureless or discontinuities) without occlusion
    """
    mask_wo_occ = mask.copy()
    occ_index = np.where(mask_occ == 128)
    mask_wo_occ[occ_index] = 255
    return mask_wo_occ


if __name__ == "__main__":
    disp_gap, discont_width = 2, 9
    textureless_width, textureless_thresh = 3, 4
    data_path = '/media/zeyun/ZEYUN/MA/middlebury-v3/'
    disp_path = 'disp_gt/'
    left_image_path = 'images/left/'

    mask_tex_path = 'mask_textureless/'
    mask_disc_path = 'mask_discont/'
    mask_occ_path = 'mask_occlusions/'
    mask_indicator_path = 'mask_indicator_wo_disc/'
    if not os.path.exists(os.path.join(data_path, mask_tex_path)):
        os.mkdir(os.path.join(data_path, mask_tex_path))
    if not os.path.exists(os.path.join(data_path, mask_disc_path)):
        os.mkdir(os.path.join(data_path, mask_disc_path))
    if not os.path.exists(os.path.join(data_path, mask_indicator_path)):
        os.mkdir(os.path.join(data_path, mask_indicator_path))

    folder = sorted(os.listdir(os.path.join(data_path, mask_occ_path)))

    for img_name in folder:
        # img_name = str(img_idx).rjust(6, '0') + '_10.png'
        gt_disparity = read(data_path + disp_path + img_name)
        left_image = read(data_path + left_image_path + img_name)

        # mask_tex = compute_textureless(left_image, textureless_width, textureless_thresh)
        # write(data_path + mask_tex_path + img_name, mask_tex)

        # mask_disc = compute_disparity_discont(gt_disparity, disp_gap, discont_width)
        # write(data_path + mask_disc_path + img_name, mask_disc)

        mask_tex = read(data_path + mask_tex_path + img_name)
        mask_disc = read(data_path + mask_disc_path + img_name)
        mask_occ = read(data_path + mask_occ_path + img_name)

        mask_indicator = compute_good_region(gt_disparity, mask_tex, None, mask_occ)
        write(data_path + mask_indicator_path + img_name, mask_indicator)