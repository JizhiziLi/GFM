<h1 align="center">Bridging Composite and Real: Towards End-to-end Deep Image Matting [IJCV-2021]</h1>

<p align="center">
  <a href="#installation">Installation</a> |
  <a href="#prepare-datasets">Prepare Datasets</a> |
  <a href="#train-on-am-2k">Train on AM-2k</a> |
  <a href="#pretrained-models">Pretrained Models</a> |
  <a href="#test-on-am-2k">Test on AM-2k</a>
</p>


## Installation
Requirements:

- Python 3.7.7+ with Numpy and scikit-image
- Pytorch (version>=1.7.1)
- Torchvision (version 0.8.2)

1. Clone this repository

    `git clone https://github.com/JizhiziLi/GFM.git`;

2. Go into the repository

    `cd GFM`;

3. Create conda environment and activate

    `conda create -n gfm python=3.7.7`,

    `conda activate gfm`;

4. Install dependencies, install pytorch and torchvision separately if you need

    `pip install -r requirements.txt`,

    `conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=10.2 -c pytorch`.

Our code has been tested with Python 3.7.7, Pytorch 1.7.1, Torchvision 0.8.2, CUDA 10.2 on Ubuntu 18.04.

## Prepare Datasets

| Dataset | <p>Dataset Link<br>(Google Drive)</p> | <p>Dataset Link<br>(Baidu Wangpan 百度网盘)</p> | Dataset Release Agreement|
| :----:| :----: | :----: | :----: | 
|<strong>AM-2k</strong>|[Link](https://drive.google.com/drive/folders/1SReB9Zma0TDfDhow7P5kiZNMwY9j9xMA?usp=sharing)|[Link](https://pan.baidu.com/s/1M1uF227-ZrYe3MBafqyTdw) (pw: 29r1)|[Agreement (MIT License)](https://jizhizili.github.io/files/gfm_datasets_agreements/AM-2k_Dataset_Release_Agreement.pdf)| 
|<strong>BG-20k</strong>|[Link](https://drive.google.com/drive/folders/1ZBaMJxZtUNHIuGj8D8v3B9Adn8dbHwSS?usp=sharing)|[Link](https://pan.baidu.com/s/1DR4uAO5j9rs-sfhn8J7oUQ) (pw: dffp)|[Agreement (MIT License)](https://jizhizili.github.io/files/gfm_datasets_agreements/BG-20k_Dataset_Release_Agreement.pdf)|

1. Download the datasets AM-2k and BG-20k from the above links and unzip to the folders `AM2K_DATASET_ROOT_PATH` and `BG20K_DATASET_ROOT_PATH`, set up the configuratures in the file `core/config.py`. Please make sure that you have checked out and agreed to the agreements.

2. For the purpose of training on ORI-Track and COMP-Track, we use the foregrounds and backgrounds generated following closed form method as in the paper `Levin, Anat, Dani Lischinski, and Yair Weiss. "A closed-form solution to natural image matting." IEEE transactions on pattern analysis and machine intelligence, 2007`. Some reference implementations can be referred to [here (python)](https://github.com/MarcoForte/closed-form-matting/blob/master/closed_form_matting/solve_foreground_background.py) and [here (matlab)](http://people.csail.mit.edu/alevin/matting.tar.gz). 

3. In order to use the composition route RSSN in COMP-Track, high-resolution backgrounds, you will need to denoise the training images in AM-2k and BG-20k. We follow the paper `Danielyan A, Katkovnik V, Egiazarian K. BM3D frames and variational image deblurring[J]. IEEE Transactions on image processing, 2011` for generating denoise images. You can refer to the [bm3d](https://pypi.org/project/bm3d/) for the installation and implementation. After denoising, make sure that you have saved the results in the folder `AM2K_DATASET_ROOT_PATH+fg_denoise/`, and `BG20K_DATASET_ROOT_PATH+train_denoise/`. We also provide the download links of these two datasets as follows for your convenience.

| Dataset | am2k_fg_denoise | bg20k_denoise |
| :----:| :----: | :----: | 
|<p>Dataset Link (Google Drive)</p> |[Link](https://drive.google.com/uc?export=download&id=1EIt4tS_ps1L5SPRLFtltj82ZNPSfHslp)|[Link](https://drive.google.com/uc?export=download&id=1SbrMH_171igViYyQpXuBgMYlNGwgbi6-)|

4. In order to use the COMP-Track on COCO datasets, you will need to download the MS COCO datasets from [here](https://cocodataset.org/) and save to path `COCO_DATASET_ROOT_PATH+train/`.

After datasets preparation, the structure of the complete datasets should be like the following. Please note that if you find the generating process inconvenience, you can refer to the following [sections](train-on-am-2k) for some <strong>easier</strong> training ways for ORI-Track and COMP-Track (no need to generate any extra datasets in advance).


```text
AM-2k
├── train
    │-- original
    │   │-- m_0a6c2018.jpg
    │   │-- ...
    │-- mask
    │   │-- m_0a6c2018.png
    │   │-- ...
    │-- fg
    │   │-- m_0a6c2018.png
    │   │-- ...
    │-- bg
    │   │-- m_0a6c2018.png
    │   │-- ...
    │-- original_denoise
    │   │-- m_0a6c2018.jpg
    │   │-- ...
    │-- fg_denoise
    │   │-- m_0a6c2018.png
    │   │-- ...
├── validation
    │-- original
    │   │-- m_0aa9dd6a.jpg
    │   │-- ...
    │── mask
    │   │-- m_0aa9dd6a.png
    │   │-- ...
    │── trimap
    │   │-- m_0aa9dd6a.png
    │   │-- ...
BG-20k
├── train
│   │-- h_0a0b8b6c.jpg
│   │-- ...
├── train_denoise
│   │-- h_0a0b8b6c.jpg
│   │-- ...
├── testval
│   │-- h_0a4a8c7f.jpg
│   │-- ...
COCO
├── train
│   │-- COCO_train2014_000000000092.jpg
│   │-- ...
```

## Train on AM-2k

### Train on the ORI-Track

Here we provide the procedure of training on ORI-Track of AM-2k:

1. Setup the environment following this [section](#installation);

2. Setup required parameters in `core/config.py`;

3.  (1) To train with closed_form foregrounds and backgrounds of AM-2k (the same way as in our paper), run the code:
    
    `chmod +x scripts/train/*`,

    `./scripts/train/train_ori.sh`;

    (2) To train <strong>easier</strong> without any extra datasets needed (no extra foregrounds and backgrounds), run the code:
    
    `chmod +x scripts/train/*`,

    `./scripts/train/train_ori_easier.sh`;

4. The training logging file will be saved in the file `logs/train_logs/args.logname`;

5. The trained model will be saved in the folder `args.model_save_dir`.


### Train on the COMP-Track

Here we provide the procedure of training on COMP-Track of AM-2k:

1. Setup the environment following this [section](#installation);

2. Setup required parameters in `core/config.py`;

3.  (1) To train with BG-20k high resolution backgrounds and RSSN composition route, and with closed_form generated foregrounds and backgrounds of AM-2k, denoised images of AM-2k and BG-20k (the same way as in our paper), run the code:
    
    `chmod +x scripts/train/*`,

    `./scripts/train/train_hd_rssn.sh`;

    (2) To train <strong>easier</strong> with BG-20k high resolution backgrounds and RSSN composition route, but not with extra closed_form foregrounds, backgrounds and the denoised images of AM-2k and BG-20k, run the code:
    
    `chmod +x scripts/train/*`,
    
    `./scripts/train/train_hd_rssn_easier.sh`;

    (2) To train with MS COCO low resolution backgrounds with closed_form foregrounds and backgrounds, run the code:
    
    `chmod +x scripts/train/*`,
    
    `./scripts/train/train_coco.sh`;

4. The training logging file will be saved in the file `logs/train_logs/args.logname`;

5. The trained model will be saved in the folder `args.model_save_dir`.


## Pretrained Models

Here we provide the models we pretrained on AM-2k ORI-Track with different backbones and RosTAs. For the backbones, `-(r)` stands for ResNet-34, `-(d)` stands for DenseNet-121, `-(r2b)` stands for ResNet-34 with 2 extra blocks, and `-(r')` stands for ResNet-101. `-TT, -FT, -BT` stand for different RosTAs.

| Model| GFM(d)-TT | GFM(r)-TT | GFM(r)-FT | GFM(r)-BT |GFM(r2b)-TT | GFM(r')-TT | GFM(d)-RIM |
| :----:| :----: | :----: | :----: |  :----: |  :----: |  :----: | :----: | 
| Google Drive |[Link](https://drive.google.com/uc?export=download&id=1knbK5uU8AitE5OFpul9FXaQm47pZO0N8)|[Link](https://drive.google.com/uc?export=download&id=1AdtoIdYTLsjXfVe_a50tin0cFwZMSz93)|[Link](https://drive.google.com/uc?export=download&id=1heWT3fodV5so--tG6hoytB11OHBfhd1_)|[Link](https://drive.google.com/uc?export=download&id=1vRWJtD8liZjb7GbNnmJCuO_xN4-bj-2l)|[Link](https://drive.google.com/uc?export=download&id=1Y8dgOprcPWdUgHUPSdue0lkFAUVvW10Q)|[Link](https://drive.google.com/uc?export=download&id=1oQMOfVkuxLujgFEQReWJM95chPlQBTiM)| [Link](https://drive.google.com/uc?export=download&id=1aMY5tJQ79IB-0NJ9WzcTu6ZOQTowiFWv) |
| <p>Baidu Wangpan<br>(百度网盘)</p> |<p><a href="https://pan.baidu.com/s/1AzuMphkNtt5-fJh-VqPnCA">Link</a><br>(pw: l6bd)</p>|<p><a href="https://pan.baidu.com/s/14TfNWeDGzXm4w91eHWu28w">Link</a><br>(pw: svcv)</p>|<p><a href="https://pan.baidu.com/s/1GmaXfiWbK09X4zhsRgooBg">Link</a><br>(pw: jfli)</p>|<p><a href="https://pan.baidu.com/s/1oaT5R8GnMW-zbbCwie1SlA">Link</a><br>(pw: 80k8)</p>|<p><a href="https://pan.baidu.com/s/1yfRGgI9QFUW9jb878AXHTg">Link</a><br>(pw: 34hf)</p>|<p><a href="https://pan.baidu.com/s/1aKUEB1MYIDbt-8iOHq67zQ">Link</a><br>(pw: 7p8j)</p>| <p><a href="https://pan.baidu.com/s/1V-YjxUsyzUsTRO8m6JoR_Q">Link</a><br>(pw: mrf7)</p>|

## Test on AM-2k

1. Create test logging folder `logs/test_logs/`;

2. Download pretrained models as shown in the previous section, unzip to the folder `models/pretrained/`;

3. Download AM-2k dataset in root `AM2K_DATASET_ROOT_PATH`;

4. Setup parameters in `scripts/test/test_dataset.sh` and run it

    `chmod +x scripts/test/*`

    `./scripts/test/test_dataset.sh`

5. The results of the alpha matte will be saved in folder `args.test_result_dir`. The logging files including the evaluation results will be saved in the file `logs/test_logs/args.logname`. Note that there may be some slight differences of the evaluation results with the ones reported in the paper due to some packages versions differences and the testing strategy. Some test results on AM-2k by the backbone `r34_2b` and the rosta `-TT` can be seen at [here](https://github.com/JizhiziLi/GFM/tree/master/demo/).