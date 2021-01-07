# EDSR-Super-Resolution

###### tags: `Selected Topics in Visual Recognition using Deep Learning` `Super Resolution`

This project is part of a series of projects for the course Selected Topics in Visual Recognition using Deep Learning, NCTU. This Repository gathers the code for image super resolution on 291 training dataset and Set14 testing dataset.

In this image super resolution task, we need to upscale the test images with an upscaling factor of 3, e.g., 120x120 -> 360x360. And we use [EDSR [1]](https://arxiv.org/pdf/1707.02921.pdf) as the main model architecture in our project.

## Environment
Framework: TensorFlow

Platform: Ubuntu (Linux)

## Reproducing Submissoin
To reproduct my submission without retrainig, do the following steps:

1. [Installation](#Installation)
2. [Download Official Dataset](#Download-Official-Dataset)
3. [Project Structure](#Project-Structure)
4. [Dataset Preparation](#Dataset-Preparation)
5. [Inference](#Inference)

## Installation

### Dependencies
- Python 3.6
- TensorFlow-gpu 1.12
- scikit-image
- pillow
- cv2

## Download Official Dataset
Download 291 traning dataset and Set14 testing dataset from [Google drive](https://drive.google.com/drive/folders/1r_vkLBCc2_d7J-rNWrgCzZUkQobvurAS?usp=sharing)

## Project Structure
```
Root/
    training_hr_images                    # 291 training dataset
    ├ 2092.png
    ├ ...
    testing_lr_images                     # Set14 testing dataset
    ├ 00.png 
    ├ ...
    results                               # HR Image in our experiment
    ├── EDSR_tf_ep100_25.602 
    │   │    ├ 00.png                     # HR Img from LR Img inferenced by EDSR model
    │   │    ├ 01.png
    │   │    ├ ...
    EDSR_Tensorflow 
    ├ main.py                             # Used for training and inferring
    ├ 2092.png
    ├── models                            # The place trained model will be save d
    │    ├ EDSRorig_x3.pb                 # trained model
    │    ├ ...
    ├── images                            # The place inferred img will be saved 
    │    ├ ...
```

## Dataset Preparation
[Download official dataset](#Download-Official-Dataset), then layout your directory and move downloaded dataset same as [Project Struture](#Project-Structure).

## Training 
Run:
```
cd EDSR_Tensorflow
python main.py --train --fromscratch --scale 3 --traindir ../training_hr_images --epochs 100 # Train from scratch, see main.py file for more optional arguments

# Export to .pb checkpoint 
python main.py --export --scale 3
```
The checkpoint(.pb) will be saved in ```EDSR_Tensorflow/models/EDSRorig_x3.pb```.

**In this competition, pretrained model are not allowed.*
## Inference
If you want to reproduct my submission without retrainig, please download our final model from [Google Drive]() and move downloaded model same as [Project Struture](#Project-Structure).

Run:
```
python main.py --upscale --scale 3 --image /path-to-image/

# For instance
python main.py --upscale --scale 3 --image ../testing_lr_images/00.png 
```
It will produce the upscale image in ```EDSR_Tensorflow/images/EdsrOutput_<img_filename>.png```

## Result

| Method | Bicubic interpolation | [VDSR [3]](https://github.com/twtygqyy/pytorch-vdsr) | [EDSR [4]](https://github.com/Saafke/EDSR_Tensorflow)
| -------- | -------- | -------- | -------- |
| PSNR     | 24.9     | 22.6     | 25.6     |

**Note: Obviously, VDSR should be better than bicubic interpolation, but in our experiment is not. After tracing the source code in [github [3]](https://github.com/twtygqyy/pytorch-vdsr), we think it's bacause the some evaluation tricks in [VDSR github code [3]](https://github.com/twtygqyy/pytorch-vdsr) we adopt.*

## Visualizer 
Low Resolutoin:

<img src="testing_lr_images/00.png" />

3x Upscale High Resolutoin

<img src="results/EDSR_tf_ep100_25.602/00.png" />

## Acknowledgements
This code is built on [VDSR (PyTorch) [3]](https://github.com/twtygqyy/pytorch-vdsr) and [EDSR (Tensorflow) [4]](https://github.com/Saafke/EDSR_Tensorflow). We thank the authors for sharing their codes.

## References
[1] [Bee Lim, Sanghyun Son, Heewon Kim, Seungjun Nah, and Kyoung Mu Lee, "Enhanced Deep Residual Networks for Single Image Super-Resolution," 2nd NTIRE: New Trends in Image Restoration and Enhancement workshop and challenge on image super-resolution in conjunction with CVPR 2017](https://arxiv.org/pdf/1707.02921.pdf)

[2] [Accurate Image Super-Resolution Using Very Deep Convolutional Networks ](https://cv.snu.ac.kr/research/VDSR/)

[3] [PyTorch VDSR](https://github.com/twtygqyy/pytorch-vdsr)

[4] [EDSR in TensorFlow](https://github.com/Saafke/EDSR_Tensorflow)



