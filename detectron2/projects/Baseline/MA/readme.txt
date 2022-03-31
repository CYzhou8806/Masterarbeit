"cityscapes_jointestamation.py",
"kitti360_jointestamation.py",
"kitti360_jointestamation_disparity.py",
"kitti_2015_dispar.py",
"kitti_2015_joint.py",
"sceneflow_ driving_dispar.py"
and "sceneflow_flying3d_dispar.py"
are all scripts for processing the dataset, which is necessary to use the corresponding dataset.

"config.py" is to adapt the detectron2 framework, with the config file in folder "configs" to set the network settings

"network" contains the main structure of the network.

"submodule" come from PSM-Net. It provides some basic units for disparityregression and stacked hourglass.

"target_generator.py", "panoptic_seg.py" and "post_processing.py" responsible for the panoptic segmentation.

"joint_evaluation.py" is for evaluating training results.