<!-- # animal-matting -->
# End-to-end Animal Image Matting
<!-- by
Author 1,
Author 2,
etc -->

<!-- ## Introduction -->

<p align="justify">This repository contains the code, datasets, models, test results and a video demo for the paper <a href="https://arxiv.org/pdf/2010.16188.pdf">End-to-end Animal Image Matting</a>. We propose a novel Glance and Focus Matting network (<strong>GFM</strong>), which employs a shared encoder and two separate decoders to learn both tasks in a collaborative manner for end-to-end animal matting. We also establish a novel Animal Matting dataset (<strong>AM-2k</strong>) to serve for end-to-end matting task. Furthermore, we investigate the domain gap issue between composition images and natural images systematically, propose a carefully designed composite route <strong>RSSN</strong> and a large-scale high-resolution background dataset (<strong>BG-20k</strong>) to serve as better candidates for composition.</p>

[Here](https://drive.google.com/file/d/1-NyeclNim9jAehrxGrbK_1PbFTgDZH5S/view?usp=sharing) is a video demo to illustrate the motivation, the network, the datasets, and the test results on an animal video.

We have released the inference code and a pretrained model, which can be found in the following sections. Since the paper is currently under review, the two datasets (**AM-2k** and **BG-20k**), training code and the rest pretrained models will be made public after review. 

## GFM
The architecture of our proposed end-to-end method <strong>GFM</strong> is illustrated below. We adopt three kinds of <em>Representation of Semantic and Transition Area</em> (<strong>RoSTa</strong>) `-TT, -FT, -BT` within our method. 
![](demo/src/gfm.png)

We trained GFM with three backbones, `-(d)` (DenseNet-121), `-(r)` (ResNet-34), and `-(r2b)` ([ResNet-34 with 2 extra blocks](core/network/e2e_resnet34_2b_gfm_tt.py)). The trained model for each backbone can be downloaded via the link listed below.


| GFM(d)-TT | GFM(r)-TT | GFM(r2b)-TT|
| :----:| :----: | :----: | 
|coming soon|coming soon|[model](https://drive.google.com/file/d/1Y8dgOprcPWdUgHUPSdue0lkFAUVvW10Q/view?usp=sharing)| 


## AM-2k
Our proposed <strong>AM-2k</strong> contains 2,000 high-resolution natural animal images from 20 categories along with manually labeled alpha mattes. Some examples are shown as below, more can be viewed in the [video demo](https://drive.google.com/file/d/1-NyeclNim9jAehrxGrbK_1PbFTgDZH5S/view?usp=sharing).
![](demo/src/am2k.png)

## BG-20k
Our proposed <strong>BG-20k</strong> contains 20,000 high-resolution background images excluded salient objects. Some examples are shown as below, more can be viewed in the [video demo](https://drive.google.com/file/d/1-NyeclNim9jAehrxGrbK_1PbFTgDZH5S/view?usp=sharing).
![](demo/src/bg20k.jpg)

## Results Demo

We test GFM on our AM-2k test dataset and show the results as below. More results on AM-2k test set can be found [here](https://github.com/JizhiziLi/animal-matting/tree/master/demo/).

<img src="demo/src/sample3.jpg" width="50%"><img src="demo/src/sample3.png" width="50%">

<img src="demo/src/sample1.jpg" width="50%"><img src="demo/src/sample1.png" width="50%">

<img src="demo/src/sample2.jpg" width="50%"><img src="demo/src/sample2.png" width="50%">

## Installation
Requirements:

- Python 3.6.5+ with Numpy and scikit-image
- Pytorch (version 1.4.0)
- Torchvision (version 0.5.0)

1. Clone this repository

    `git clone https://github.com/JizhiziLi/animal-matting.git`

2. Go into the repository

    `cd animal-matting`

3. Create conda environment and activate

    `conda create -n animalmatting python=3.6.5`

    `conda activate animalmatting`

4. Install dependencies, install pytorch and torchvision separately if you need

    `pip install -r requirements.txt`

    `conda install pytorch==1.4.0 torchvision==0.5.0 cudatoolkit=10.1 -c pytorch`

Our code has been tested with Python 3.6.5, Pytorch 1.4.0, Torchvision 0.5.0, CUDA 10.1 on Ubuntu 18.04.


## Inference Code - How to Test on Your Images

Here we provide the procedure of testing on sample images by our pretrained models:

1. Download pretrained models as shown in section **GFM**, unzip to folder `models/`

2. Save your sample images in folder `samples/original/.`

3. Setup parameters in `scripts/test_samples.sh` and run it

    `chmod +x scripts/*`

    `./scripts/test_samples.sh`

4. The results of alpha matte and transparent color image will be saved in folder `samples/result_alpha/.` and `samples/result_color/.`


We show some sample images from the internet, the predicted alpha mattes, and their transparent results as below. *(We adopt arch='[e2e_resnet34_2b_gfm_tt](https://drive.google.com/file/d/1Y8dgOprcPWdUgHUPSdue0lkFAUVvW10Q/view?usp=sharing)' and use hybrid testing strategy.)*

<img src="samples/original/1.jpg" width="33%"><img src="samples/result_alpha/1.png" width="33%"><img src="samples/result_color/1.png" width="33%">
<img src="samples/original/2.jpg" width="33%"><img src="samples/result_alpha/2.png" width="33%"><img src="samples/result_color/2.png" width="33%">
<img src="samples/original/3.jpg" width="33%"><img src="samples/result_alpha/3.png" width="33%"><img src="samples/result_color/3.png" width="33%">



## Statement
This project is for research purpose only, please contact us for the licence of commercial use. For any other questions please contact [jili8515@uni.sydney.edu.au](mailto:jili8515@uni.sydney.edu.au).

