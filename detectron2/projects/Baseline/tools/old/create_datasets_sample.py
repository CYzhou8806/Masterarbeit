import os
import shutil

os.environ['KITTI360_DATASET'] = os.path.join("/bigwork/nhgnycao/Masterarbeit/detectron2/projects/Baseline/datasets", "kitti_360")
output_root = os.path.join(os.environ['KITTI360_DATASET'], "train")

count = 0
for root, dirs, files in os.walk(os.path.join(output_root,"disparit")):
    for file in files:
        if os.path.splitext(file)[-1] == '.tiff':
            disparity_path = os.path.join(root,file)
            basename = os.path.basename(disparity_path)
            file_id = os.path.splitext(basename)[0]

            left_path = os.path.join(root, file_id).replace("disparity", "left") + '.png'
            left_path = left_path.replace("disparit", "lef")

            right_path = os.path.join(root, file_id).replace("disparity", "right") + '.png'
            right_path = right_path.replace("disparit", "righ")

            shutil.copy(disparity_path, os.path.join(output_root,"disparity",os.path.basename(disparity_path)))
            shutil.copy(left_path, os.path.join(output_root,"left",os.path.basename(left_path)))
            shutil.copy(right_path, os.path.join(output_root,"right",os.path.basename(right_path)))

            count +=1
        if count >=200:
            raise RuntimeError("excepted stop")
