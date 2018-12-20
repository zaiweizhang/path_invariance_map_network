# Path Invariance Network in Tensorflow

Tensorflow implementation of one application in Path Invariance Map Networks: 3D Semantic Segmentation.
Since we use [pointnet2](https://github.com/charlesq34/pointnet2) codebase, some of the code are borrowed from there.

![alt tag](intro.JPG)

## Prerequisites

- Python 2.7 
- [Tensorflow 0.12.1](https://github.com/tensorflow/tensorflow/tree/r0.12)
- [SciPy](http://www.scipy.org/install.html)
- [Pointnet2 Customized TF Operators](https://github.com/charlesq34/pointnet2/tree/master/tf_ops) Please make sure the tf_ops folder is placed in the same directory.

## Usage

First, download dataset [here](). (Point clouds are collected by Pointnet++)
The pre-trained models can be downloaded [here]().
(Point clouds models are trained using Pointnet++, and voxel models are trained using 3D-U-Net. The training code for voxel models will be released soon.)

To train models with downloaded dataset:

    $ ./cmd

All training commands are included in the cmd file.
Testing results are logged during the training stage.

## Results
![alt tag](result.JPG)

## Author

Zaiwei Zhang
