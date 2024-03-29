#!/usr/bin/python
# -*- coding: utf-8 -*-


#################
## Import modules
#################
import sys
# walk directories
import glob
# access to OS functionality
import os
# copy things
import copy
# numpy
import numpy as np
# open3d
# import open3d
# matplotlib for colormaps
import matplotlib.cm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# struct for reading binary ply files
import struct
import shutil
from tqdm import tqdm

os.environ["KITTI360_DATASET"] = "/media/eistrauben/Dinge/Masterarbeit/dataset/kitti_360"
#os.environ["KITTI360_DATASET"] = "/bigwork/nhgnycao/datasets/KITTI-360"
#os.environ["KITTI360_DATASET"] = r"D:\Masterarbeit\dataset\kitti_360"


# the main class that loads raw 3D scans
class Kitti360Viewer3DRaw(object):

    # Constructor
    def __init__(self, seq=0, mode='velodyne'):

        if 'KITTI360_DATASET' in os.environ:
            kitti360Path = os.environ['KITTI360_DATASET']
        else:
            kitti360Path = os.path.join(os.path.dirname(
                os.path.realpath(__file__)), '..', '..')

        if mode == 'velodyne':
            self.sensor_dir = 'velodyne_points'
        elif mode == 'sick':
            self.sensor_dir = 'sick_points'
        else:
            raise RuntimeError('Unknown sensor type!')

        sequence = '2013_05_28_drive_%04d_sync' % seq
        self.raw3DPcdPath = os.path.join(kitti360Path, 'data_3d_raw', sequence, self.sensor_dir, 'data')

    def loadVelodyneData(self, frame=0):
        pcdFile = os.path.join(self.raw3DPcdPath, '%010d.bin' % frame)
        if not os.path.isfile(pcdFile):
            raise RuntimeError('%s does not exist!' % pcdFile)
        pcd = np.fromfile(pcdFile, dtype=np.float32)
        pcd = np.reshape(pcd, [-1, 4])
        return pcd

    def loadSickData(self, frame=0):
        pcdFile = os.path.join(self.raw3DPcdPath, '%010d.bin' % frame)
        if not os.path.isfile(pcdFile):
            raise RuntimeError('%s does not exist!' % pcdFile)
        pcd = np.fromfile(pcdFile, dtype=np.float32)
        pcd = np.reshape(pcd, [-1, 2])
        pcd = np.concatenate([np.zeros_like(pcd[:, 0:1]), -pcd[:, 0:1], pcd[:, 1:2]], axis=1)
        return pcd


def projectVeloToImage(output_root, cam_id=0, seq=0):
    from kitti360scripts.devkits.commons.loadCalibration import loadCalibrationCameraToPose, loadCalibrationRigid
    from kitti360scripts.helpers.project import CameraPerspective, CameraFisheye
    from PIL import Image
    import matplotlib.pyplot as plt

    if 'KITTI360_DATASET' in os.environ:
        kitti360Path = os.environ['KITTI360_DATASET']
    else:
        kitti360Path = os.path.join(os.path.dirname(
            os.path.realpath(__file__)), '..', '..')

    disp_gt_save_dir = os.path.join(output_root, "disparity")
    left_save_dir = os.path.join(output_root, "left")
    right_save_dir = os.path.join(output_root, "right")
    if not os.path.exists(disp_gt_save_dir):
        os.makedirs(disp_gt_save_dir)
    if not os.path.exists(left_save_dir):
        os.makedirs(left_save_dir)
    if not os.path.exists(right_save_dir):
        os.makedirs(right_save_dir)


    baseline = 600.0  # 0.60 m
    # take fx of the to projected image
    if cam_id == 0:
        f = 552.554261  # f = 788.629315    # 552.554261
    elif cam_id == 1:
        f = 785.134093
    else:
        raise ValueError("Up to now only Perspective Camera supported")
    depth_disp_konst = baseline * f

    sequence = '2013_05_28_drive_%04d_sync' % seq

    '''
    output_root = os.path.join(kitti360Path, "disparity")
    if not os.path.exists(output_root):
        os.makedirs(output_root)
        print("---  create new folder...  ---")
    else:
        shutil.rmtree(output_root)  # 递归删除文件夹
        os.makedirs(output_root)
        print("---  del old and create new folder...  ---")
    '''

    # perspective camera
    if cam_id in [0, 1]:
        camera = CameraPerspective(kitti360Path, sequence, cam_id)
    # fisheye camera
    elif cam_id in [2, 3]:
        camera = CameraFisheye(kitti360Path, sequence, cam_id)
    else:
        raise RuntimeError('Unknown camera ID!')

    # object for parsing 3d raw data 
    velo = Kitti360Viewer3DRaw(mode='velodyne', seq=seq)
    sick = Kitti360Viewer3DRaw(mode='sick', seq=seq)

    # cam_0 to velo
    fileCameraToVelo = os.path.join(kitti360Path, 'calibration', 'calib_cam_to_velo.txt')
    TrCam0ToVelo = loadCalibrationRigid(fileCameraToVelo)

    # sick to velo
    fileSickToVelo = os.path.join(kitti360Path, 'calibration', 'calib_sick_to_velo.txt')
    TrSickToVelo = loadCalibrationRigid(fileSickToVelo)

    # all cameras to system center 
    fileCameraToPose = os.path.join(kitti360Path, 'calibration', 'calib_cam_to_pose.txt')
    TrCamToPose = loadCalibrationCameraToPose(fileCameraToPose)

    # velodyne to all cameras
    TrVeloToCam = {}
    TrSickToCam = {}
    for k, v in TrCamToPose.items():
        # Tr(cam_k -> velo) = Tr(cam_k -> cam_0) @ Tr(cam_0 -> velo)
        TrCamkToCam0 = np.linalg.inv(TrCamToPose['image_00']) @ TrCamToPose[k]
        TrCamToVelo = TrCam0ToVelo @ TrCamkToCam0
        # Tr(velo -> cam_k)
        TrVeloToCam[k] = np.linalg.inv(TrCamToVelo)

        # Tr(cam_k -> sick) = Tr(cam_k -> velo) @ Tr(velo -> sick)
        TrVeloToSick = np.linalg.inv(TrSickToVelo)
        TrCamToSick = TrVeloToSick @ TrCamToVelo
        # Tr(sick -> cam_k)
        TrSickToCam[k] = np.linalg.inv(TrCamToSick)

    # take the rectification into account for perspective cameras
    if cam_id == 0 or cam_id == 1:
        TrVeloToRect = np.matmul(camera.R_rect, TrVeloToCam['image_%02d' % cam_id])
        TrSickToRect = np.matmul(camera.R_rect, TrSickToCam['image_%02d' % cam_id])
    else:
        TrVeloToRect = TrVeloToCam['image_%02d' % cam_id]
        TrSickToRect = TrSickToCam['image_%02d' % cam_id]

    # color map for visualizing depth map
    cm = plt.get_cmap('jet')

    count = 0
    # visualize a set of frame
    # for each frame, load the raw 3D scan and project to image plane
    # for frame in tqdm(range(0, 1000, 2)):
    sub_dir = 'data_rect' if cam_id in [0, 1] else 'data_rgb'
    for root, dirs, files in os.walk(os.path.join(kitti360Path, 'data_2d_raw', sequence, 'image_%02d' % cam_id, sub_dir)):
        for file in tqdm(files):
            if os.path.splitext(file)[-1] == '.png':
                count +=1
                frame = int(os.path.splitext(file)[0])

                ## velo
                points = velo.loadVelodyneData(frame)
                points[:, 3] = 1

                # transfrom velodyne points to camera coordinate
                pointsCam = np.matmul(TrVeloToRect, points.T).T
                pointsCam = pointsCam[:, :3]
                # project to image space
                u, v, depth = camera.cam2image(pointsCam.T)
                u = u.astype(np.int64)
                v = v.astype(np.int64)

                # prepare depth map for visualization
                depthMap = np.zeros((camera.height, camera.width))
                depthImage = np.zeros((camera.height, camera.width, 3))
                mask = np.logical_and(np.logical_and(np.logical_and(u >= 0, u < camera.width), v >= 0), v < camera.height)
                # visualize points within 30 meters
                mask = np.logical_and(np.logical_and(mask, depth > 0), depth < 99999)
                depthMap[v[mask], u[mask]] = depth[mask]

                sub_dir = 'data_rect' if cam_id in [0, 1] else 'data_rgb'
                imagePath = os.path.join(kitti360Path, 'data_2d_raw', sequence, 'image_%02d' % cam_id, sub_dir,
                                         '%010d.png' % frame)
                if not os.path.isfile(imagePath):
                    raise RuntimeError('Image file %s does not exist!' % imagePath)

                img_name = os.path.basename(imagePath)

                # depth to disparity
                mask_nozero = depthMap != 0
                dispMap_velo = np.zeros_like(depthMap)
                dispMap_velo[mask_nozero] = depth_disp_konst / (depthMap[mask_nozero] * 1000)
                # dispMap_velo[mask_nozero] = (depthMap[mask_nozero])

                '''
                ## sick
                points = sick.loadSickData(frame)
                points = np.concatenate([points, points[:, 1:2]], axis=1)
                points[:, 3] = 1

                # transfrom sick points to camera coordinate
                pointsCam = np.matmul(TrSickToRect, points.T).T
                pointsCam = pointsCam[:, :3]
                # project to image space
                u, v, depth = camera.cam2image(pointsCam.T)
                u = u.astype(np.int64)
                v = v.astype(np.int64)

                # prepare depth map for visualization
                depthMap = np.zeros((camera.height, camera.width))
                depthImage = np.zeros((camera.height, camera.width, 3))
                mask = np.logical_and(np.logical_and(np.logical_and(u >= 0, u < camera.width), v >= 0),
                                      v < camera.height)
                # visualize points within 30 meters
                mask = np.logical_and(np.logical_and(mask, depth > 0), depth < 99999)
                depthMap[v[mask], u[mask]] = depth[mask]

                sub_dir = 'data_rect' if cam_id in [0, 1] else 'data_rgb'
                imagePath = os.path.join(kitti360Path, 'data_2d_raw', sequence, 'image_%02d' % cam_id, sub_dir,
                                         '%010d.png' % frame)
                if not os.path.isfile(imagePath):
                    raise RuntimeError('Image file %s does not exist!' % imagePath)

                img_name = os.path.basename(imagePath)

                # depth to disparity
                mask_nozero = depthMap != 0
                # dispMap_sick = np.zeros_like(depthMap)
                # dispMap_sick[mask_nozero] = depth_disp_konst / (depthMap[mask_nozero] * 1000)
                # dispMap_sick[mask_nozero] = (depthMap[mask_nozero])
                print(np.sum(dispMap_velo!=0.0))
                dispMap_velo[mask_nozero] = depth_disp_konst / (depthMap[mask_nozero] * 1000)
                print(np.sum(dispMap_velo!=0.0))
                '''
                dispMap = dispMap_velo

                # dispMap = dispMap.astype('int32')
                gt = Image.fromarray(dispMap)

                new_name_disp_gt = sequence + '_' + os.path.splitext(img_name)[0] + '_disparity.tiff'
                # new_name_disp_gt = sequence + '_' + os.path.splitext(img_name)[0] + '_disparity.png'
                gt.save(os.path.join(disp_gt_save_dir, new_name_disp_gt))

                '''
                tmp = Image.open(os.path.join(disp_gt_save_dir, new_name_disp_gt))
                tmp = np.array(tmp)
                tmp = tmp.astype(float) / 256
                print(np.all(tmp == dispMap))
                '''


                # copy raw 2D
                left_img_path = imagePath
                right_img_path = left_img_path.replace('image_00', 'image_01')
                new_name_left_img = sequence + '_' + os.path.splitext(img_name)[0] + '_left.png'
                new_name_right_img = sequence + '_' + os.path.splitext(img_name)[0] + '_right.png'
                shutil.copyfile(left_img_path, os.path.join(left_save_dir, new_name_left_img))
                shutil.copyfile(right_img_path, os.path.join(right_save_dir, new_name_right_img))

                if count >= 50:
                    return 0
                continue
                '''
                layout = (2,1) if cam_id in [0,1] else (1,2)
                sub_dir = 'data_rect' if cam_id in [0,1] else 'data_rgb'
                fig, axs = plt.subplots(*layout, figsize=(18,12))
        
                # load RGB image for visualization
                imagePath = os.path.join(kitti360Path, 'data_2d_raw', sequence, 'image_%02d' % cam_id, sub_dir, '%010d.png' % frame)
                if not os.path.isfile(imagePath):
                    raise RuntimeError('Image file %s does not exist!' % imagePath)
        
                colorImage = np.array(Image.open(imagePath)) / 255.
                depthImage = cm(depthMap/depthMap.max())[...,:3]
                colorImage[depthMap>0] = depthImage[depthMap>0]
        
                axs[0].imshow(depthMap, cmap='jet')
                axs[0].title.set_text('Projected Depth')
                axs[0].axis('off')
                axs[1].imshow(colorImage)
                axs[1].title.set_text('Projected Depth Overlaid on Image')
                axs[1].axis('off')
                plt.suptitle('Sequence %04d, Camera %02d, Frame %010d' % (seq, cam_id, frame))
                plt.show()
                '''


def projectVeloToImage_singel(cam_id=0, seq=6, ):
    from kitti360scripts.devkits.commons.loadCalibration import loadCalibrationCameraToPose, loadCalibrationRigid
    from kitti360scripts.helpers.project import CameraPerspective, CameraFisheye
    from PIL import Image
    import matplotlib.pyplot as plt

    os.environ["KITTI360_DATASET"] = r"D:\Masterarbeit\dataset\kitti_360"

    if 'KITTI360_DATASET' in os.environ:
        kitti360Path = os.environ['KITTI360_DATASET']
    else:
        kitti360Path = os.path.join(os.path.dirname(
            os.path.realpath(__file__)), '..', '..')
    kitti360Path = r"D:\Masterarbeit\dataset\kitti_360"
    baseline = 600.0  # 0.60 m
    # take fx of the to projected image
    if cam_id == 0:
        f = 552.554261  # f = 788.629315    # 552.554261
    elif cam_id == 1:
        f = 785.134093
    else:
        raise ValueError("Up to now only Perspective Camera supported")
    depth_disp_konst = baseline * f

    sequence = '2013_05_28_drive_%04d_sync' % seq

    '''
    output_root = os.path.join(kitti360Path, "disparity")
    if not os.path.exists(output_root):
        os.makedirs(output_root)
        print("---  create new folder...  ---")
    else:
        shutil.rmtree(output_root)  # 递归删除文件夹
        os.makedirs(output_root)
        print("---  del old and create new folder...  ---")
    '''

    # perspective camera
    if cam_id in [0, 1]:
        camera = CameraPerspective(kitti360Path, sequence, cam_id)
    # fisheye camera
    elif cam_id in [2, 3]:
        camera = CameraFisheye(kitti360Path, sequence, cam_id)
    else:
        raise RuntimeError('Unknown camera ID!')

    # object for parsing 3d raw data
    velo = Kitti360Viewer3DRaw(mode='velodyne', seq=seq)

    # cam_0 to velo
    fileCameraToVelo = os.path.join(kitti360Path, 'calibration', 'calib_cam_to_velo.txt')
    TrCam0ToVelo = loadCalibrationRigid(fileCameraToVelo)

    # sick to velo
    fileSickToVelo = os.path.join(kitti360Path, 'calibration', 'calib_sick_to_velo.txt')
    TrSickToVelo = loadCalibrationRigid(fileSickToVelo)

    # all cameras to system center
    fileCameraToPose = os.path.join(kitti360Path, 'calibration', 'calib_cam_to_pose.txt')
    TrCamToPose = loadCalibrationCameraToPose(fileCameraToPose)

    # velodyne to all cameras
    TrVeloToCam = {}
    TrSickToCam = {}
    for k, v in TrCamToPose.items():
        # Tr(cam_k -> velo) = Tr(cam_k -> cam_0) @ Tr(cam_0 -> velo)
        TrCamkToCam0 = np.linalg.inv(TrCamToPose['image_00']) @ TrCamToPose[k]
        TrCamToVelo = TrCam0ToVelo @ TrCamkToCam0
        # Tr(velo -> cam_k)
        TrVeloToCam[k] = np.linalg.inv(TrCamToVelo)

        # Tr(cam_k -> sick) = Tr(cam_k -> velo) @ Tr(velo -> sick)
        TrVeloToSick = np.linalg.inv(TrSickToVelo)
        TrCamToSick = TrVeloToSick @ TrCamToVelo
        # Tr(sick -> cam_k)
        TrSickToCam[k] = np.linalg.inv(TrCamToSick)

    # take the rectification into account for perspective cameras
    if cam_id == 0 or cam_id == 1:
        TrVeloToRect = np.matmul(camera.R_rect, TrVeloToCam['image_%02d' % cam_id])
    else:
        TrVeloToRect = TrVeloToCam['image_%02d' % cam_id]
        TrSickToRect = TrSickToCam['image_%02d' % cam_id]

    # color map for visualizing depth map
    cm = plt.get_cmap('jet')

    # visualize a set of frame
    # for each frame, load the raw 3D scan and project to image plane
    # for frame in tqdm(range(0, 1000, 2)):
    sub_dir = 'data_rect' if cam_id in [0, 1] else 'data_rgb'
    #for root, dirs, files in os.walk(
    #        os.path.join(kitti360Path, 'data_2d_raw', sequence, 'image_%02d' % cam_id, sub_dir)):
    #    for file in tqdm(files):
    #        if os.path.splitext(file)[-1] == '.png':
    frame = int('0000004759')

    ## velo
    points = velo.loadVelodyneData(frame)
    points[:, 3] = 1

    # transfrom velodyne points to camera coordinate
    pointsCam = np.matmul(TrVeloToRect, points.T).T
    pointsCam = pointsCam[:, :3]
    # project to image space
    u, v, depth = camera.cam2image(pointsCam.T)
    u = u.astype(np.int64)
    v = v.astype(np.int64)

    # prepare depth map for visualization
    depthMap = np.zeros((camera.height, camera.width))
    depthImage = np.zeros((camera.height, camera.width, 3))
    mask = np.logical_and(np.logical_and(np.logical_and(u >= 0, u < camera.width), v >= 0),
                          v < camera.height)
    # visualize points within 30 meters
    mask = np.logical_and(np.logical_and(mask, depth > 0), depth < 99999)
    depthMap[v[mask], u[mask]] = depth[mask]

    img_name = '%010d.png' % frame

    # depth to disparity
    mask_nozero = depthMap != 0
    dispMap_velo = np.zeros_like(depthMap)
    dispMap_velo[mask_nozero] = depth_disp_konst / (depthMap[mask_nozero] * 1000)
    # dispMap_velo[mask_nozero] = (depthMap[mask_nozero])

    '''
    ## sick
    points = sick.loadSickData(frame)
    points = np.concatenate([points, points[:, 1:2]], axis=1)
    points[:, 3] = 1

    # transfrom sick points to camera coordinate
    pointsCam = np.matmul(TrSickToRect, points.T).T
    pointsCam = pointsCam[:, :3]
    # project to image space
    u, v, depth = camera.cam2image(pointsCam.T)
    u = u.astype(np.int64)
    v = v.astype(np.int64)

    # prepare depth map for visualization
    depthMap = np.zeros((camera.height, camera.width))
    depthImage = np.zeros((camera.height, camera.width, 3))
    mask = np.logical_and(np.logical_and(np.logical_and(u >= 0, u < camera.width), v >= 0),
                          v < camera.height)
    # visualize points within 30 meters
    mask = np.logical_and(np.logical_and(mask, depth > 0), depth < 99999)
    depthMap[v[mask], u[mask]] = depth[mask]

    sub_dir = 'data_rect' if cam_id in [0, 1] else 'data_rgb'
    imagePath = os.path.join(kitti360Path, 'data_2d_raw', sequence, 'image_%02d' % cam_id, sub_dir,
                             '%010d.png' % frame)
    if not os.path.isfile(imagePath):
        raise RuntimeError('Image file %s does not exist!' % imagePath)

    img_name = os.path.basename(imagePath)

    # depth to disparity
    mask_nozero = depthMap != 0
    # dispMap_sick = np.zeros_like(depthMap)
    # dispMap_sick[mask_nozero] = depth_disp_konst / (depthMap[mask_nozero] * 1000)
    # dispMap_sick[mask_nozero] = (depthMap[mask_nozero])
    print(np.sum(dispMap_velo!=0.0))
    dispMap_velo[mask_nozero] = depth_disp_konst / (depthMap[mask_nozero] * 1000)
    print(np.sum(dispMap_velo!=0.0))
    '''
    dispMap = dispMap_velo

    # dispMap = dispMap.astype('int32')
    gt = Image.fromarray(dispMap)

    new_name_disp_gt = sequence + '_' + os.path.splitext(img_name)[0] + '_disparity.tiff'
    # new_name_disp_gt = sequence + '_' + os.path.splitext(img_name)[0] + '_disparity.png'
    gt.save(os.path.join(r'C:\Users\cyzho\Desktop', new_name_disp_gt))

    '''
    tmp = Image.open(os.path.join(disp_gt_save_dir, new_name_disp_gt))
    tmp = np.array(tmp)
    tmp = tmp.astype(float) / 256
    print(np.all(tmp == dispMap))
    '''

    '''
    # copy raw 2D
    left_img_path = imagePath
    right_img_path = left_img_path.replace('image_00', 'image_01')
    new_name_left_img = sequence + '_' + os.path.splitext(img_name)[0] + '_left.png'
    new_name_right_img = sequence + '_' + os.path.splitext(img_name)[0] + '_right.png'
    shutil.copyfile(left_img_path, os.path.join(left_save_dir, new_name_left_img))
    shutil.copyfile(right_img_path, os.path.join(right_save_dir, new_name_right_img))
    '''

    '''
    layout = (2,1) if cam_id in [0,1] else (1,2)
    sub_dir = 'data_rect' if cam_id in [0,1] else 'data_rgb'
    fig, axs = plt.subplots(*layout, figsize=(18,12))

    # load RGB image for visualization
    imagePath = os.path.join(kitti360Path, 'data_2d_raw', sequence, 'image_%02d' % cam_id, sub_dir, '%010d.png' % frame)
    if not os.path.isfile(imagePath):
        raise RuntimeError('Image file %s does not exist!' % imagePath)

    colorImage = np.array(Image.open(imagePath)) / 255.
    depthImage = cm(depthMap/depthMap.max())[...,:3]
    colorImage[depthMap>0] = depthImage[depthMap>0]

    axs[0].imshow(depthMap, cmap='jet')
    axs[0].title.set_text('Projected Depth')
    axs[0].axis('off')
    axs[1].imshow(colorImage)
    axs[1].title.set_text('Projected Depth Overlaid on Image')
    axs[1].axis('off')
    plt.suptitle('Sequence %04d, Camera %02d, Frame %010d' % (seq, cam_id, frame))
    plt.show()
    '''


if __name__ == '__main__':
    if 'KITTI360_DATASET' in os.environ:
        kitti360Path = os.environ['KITTI360_DATASET']
    else:
        kitti360Path = os.path.join(os.path.dirname(
            os.path.realpath(__file__)), '..', '..')

    output_root = os.path.join(kitti360Path, "kitti_360_demo")
    #output_root = os.path.join("/bigwork/nhgnycao/Masterarbeit/detectron2/projects/Baseline/datasets", "kitti_360")
    if not os.path.exists(output_root):
        os.makedirs(output_root)
        print("---  create new folder...  ---")
    #else:
    #    shutil.rmtree(output_root)  # 递归删除文件夹
    #    os.makedirs(output_root)
    #    print("---  del old and create new folder...  ---")

    visualizeIn2D = True
    # sequence index
    seq = 0
    # set it to 0 or 1 for projection to perspective images
    #           2 or 3 for projecting to fisheye images
    cam_id = 0

    train_dir = os.path.join(output_root, "train")
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    '''
    for seq in [0, 3, 5, 6, 7, 10]:
        # visualize raw 3D velodyne scans in 2D
        if visualizeIn2D:
            projectVeloToImage(train_dir, seq=seq, cam_id=cam_id)

    test_dir = os.path.join(output_root, "test")
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)
    for seq in [2, 4, 9]:
        # visualize raw 3D velodyne scans in 2D
        if visualizeIn2D:
            projectVeloToImage(test_dir, seq=seq, cam_id=cam_id)

    '''
    if visualizeIn2D:
        projectVeloToImage(train_dir, seq=seq, cam_id=cam_id)


    #projectVeloToImage_singel()


