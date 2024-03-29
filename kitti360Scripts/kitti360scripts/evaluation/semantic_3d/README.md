###########################################################################

                KITTI-360: THE 3D SEMANTIC SEGMENATION BENCHMARK          
                     Yiyi Liao     Jun Xie     Andreas Geiger             
                            University of Tübingen                        
           Max Planck Institute for Intelligent Systems, Tübingen         
                      www.cvlibs.net/datasets/kitti-360                   

###########################################################################



This file describes the KITTI-360 3D semantic segmentation benchmark that consists of 42 test windows. 


## Train/Val Data ##

The accumulated point clouds for training and validation as well as the training/validation split can be found in 
```
Download -> 3D data & labels -> Accumulated Point Cloud for Train & Val (128G)
```
Note that the accumulated point clouds are released as a set of batches which we refer to as `windows`. A single window contains observations within a driving distance of about 200 meters (240 frames on average) and there is an overlap of 10 meters between two consecutive windows. We provide the following information for each 3D point of the accumulated point cloud:
```
- float x 
- float y
- float z
- uchar red
- uchar green
- uchar blue
- int semantic
- int instance
- uchar visible
- float confidence
```
where `x y z` is the location of a 3D point in the world coordinate, `red green blue` is the color of a 3D point obtained by projecting it to adjacent 2D images, `semantic instance` describes the label of a 3D point. `visible` is 0 if not visible in any of the 2D frames and 1 otherwise. `confidence` is the confidence of the 3D label. Please refer to the "3D Data Format" of our online [documentation](http://www.cvlibs.net/datasets/kitti-360/documentation.php) for more details of the data format. 

The training and validation split can be found in TODO


## Test Data ##

We evaluate on 42 windows from two test sequences, 0008 and 0018. The accumulated point cloud of the test windows can be found in:
```
Download -> 3D data & labels -> Test (1.2G)
```
We provide the following information for the test windows:
```
- float x 
- float y
- float z
- uchar red
- uchar green
- uchar blue
- uchar visible
```
Compared to the training & validation windows, `semantic`, `instance` and `confidence` are held out on the evaluation server.


## Output for 3D Semantic Segmentation ##

The output structure should be analogous to the input.
All results must be provided in the root directory of a zip file using the format of npy. The file names should follow `{seq:0>4}_{start_frame:0>10}_{end_frame:0>10}.npy`. Here is how the semantic predictions should look like in root directory of your zip file. 
```
0008_0000000002_0000000245.npy
0008_0000000235_0000000608.npy
...
0018_0000000002_0000000341.npy
0018_0000000330_0000000543.npy
...
```
Each numpy file should contain a vector of the semantic labels corresponding to the accumulated point cloud. Given `N` points in the accumulated point cloud, the submitted .npy file should contain __only__ a vector of length `N`, where the `i`th scalar indicates the semantic label of the `i`th point in the accumulated point cloud. 

The semantic labels should follow the definition of [labels.py](https://github.com/autonomousvision/kitti360Scripts/blob/master/kitti360scripts/helpers/labels.py). Note that `id` should be used instead of `kittiId` or `trainId`.
