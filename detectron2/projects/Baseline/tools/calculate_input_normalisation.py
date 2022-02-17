import os

import numpy as np
from PIL import Image, ImageFilter
from tqdm import tqdm
from prettytable import PrettyTable


def calculate_mean_std(datasets_dic, ):
    # calculate mean
    count = 0
    to_calculate_mean = []
    for dataset_name, left_root in datasets_dic.items():
        for root, dirs, files in os.walk(left_root):
            for file in tqdm(files):
                if os.path.splitext(file)[-1] == '.png':
                    left_path = os.path.join(root, file)

                    left_img = Image.open(left_path)
                    left_np = np.array(left_img)
                    if dataset_name[:2] == 'SF':
                        left_np = np.delete(left_np, -1, axis=2)
                    assert left_np.shape[2] == 3
                    cur_mean = []
                    count += left_np.shape[0] * left_np.shape[1]  # h * w
                    for i in range(left_np.shape[2]):
                        cur_mean.append(np.mean(left_np[:, :, i]).astype(float))
                    cur_mean.append(left_np.shape[0] * left_np.shape[1])
                    to_calculate_mean.append(cur_mean)
    channel0 = 0
    channel1 = 0
    channel2 = 0
    for single_data in to_calculate_mean:
        channel0 += single_data[0] * single_data[-1]
        channel1 += single_data[1] * single_data[-1]
        channel2 += single_data[2] * single_data[-1]
    channel1_mean = channel0 / count
    channel2_mean = channel1 / count
    channel3_mean = channel2 / count
    mean_res = [channel1_mean, channel2_mean, channel3_mean]

    # calculate std
    variance_sum = [0, 0, 0]
    for dataset_name, left_root in datasets_dic.items():
        for root, dirs, files in os.walk(left_root):
            for file in tqdm(files):
                if os.path.splitext(file)[-1] == '.png':
                    left_path = os.path.join(root, file)
                    left_img = Image.open(left_path)
                    left_np = np.array(left_img).astype(float)
                    if dataset_name[:2] == 'SF':
                        left_np = np.delete(left_np, -1, axis=2)

                    left_np = left_np.transpose(2, 0, 1)
                    assert left_np.shape[0] == 3
                    for i in range(left_np.shape[0]):
                        left_np[i, :, :] = left_np[i, :, :] - mean_res[i]
                        variance_sum[i] += np.sum(left_np[i, :, :] ** 2)

    variance = np.array(variance_sum).astype(float) / count
    std = np.sqrt(variance)

    return mean_res, list(std)


if __name__ == '__main__':
    datasets = {
        'SF_driving_train': '/home/eistrauben/github/Masterarbeit/detectron2/projects/Baseline/datasets/sceneflow/driving/train/left',
        'SF_driving_val': '/home/eistrauben/github/Masterarbeit/detectron2/projects/Baseline/datasets/sceneflow/driving/val/left',
        'kitti2015_train': '/home/eistrauben/github/Masterarbeit/detectron2/projects/Baseline/datasets/kitti_2015/data_scene_flow/training/image_2',
        'kitti2015_val': '/home/eistrauben/github/Masterarbeit/detectron2/projects/Baseline/datasets/kitti_2015/data_scene_flow/val/image_2',
        'kitti2015_test': '/home/eistrauben/github/Masterarbeit/detectron2/projects/Baseline/datasets/kitti_2015/data_scene_flow/test/image_2',
    }

    datasets = {
        'SF_flying3d_train': '/media/eistrauben/移动胡萝卜框/dataset/flying3d/train/left',
        'SF_flying3d_val': '/media/eistrauben/移动胡萝卜框/dataset/flying3d/val/left',
        'kitti2015_train': '/home/eistrauben/github/Masterarbeit/detectron2/projects/Baseline/datasets/kitti_2015/data_scene_flow/training/image_2',
        'kitti2015_val': '/home/eistrauben/github/Masterarbeit/detectron2/projects/Baseline/datasets/kitti_2015/data_scene_flow/val/image_2',
        'kitti2015_test': '/home/eistrauben/github/Masterarbeit/detectron2/projects/Baseline/datasets/kitti_2015/data_scene_flow/test/image_2',
    }

    # create details table
    tabel_detail = PrettyTable(['dataset', 'mean', 'std', ])
    for dataset_name, path in tqdm(datasets.items()):
        mean, std = calculate_mean_std({dataset_name: path, })
        tabel_detail.add_row([dataset_name, str(mean), str(std)])

    mean, std = calculate_mean_std(datasets)
    tabel_detail.add_row(['all', str(mean), str(std)])

    print(tabel_detail)
    print('\n\n')

    f = open('datasets_normalisation.txt', "w")
    tabel_detail_txt = tabel_detail.get_string()
    f.write(tabel_detail_txt)
    f.write('\n\n\n')
    f.close()

