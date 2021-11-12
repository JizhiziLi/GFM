#/bin/bash
## [1]
## Need to generate the closed_form foregrounds and backgrounds of AM-2K
## following the paper "A closed-form solution to natural image matting." IEEE transactions on pattern analysis and machine intelligence, 2007
## and save to path DATASET_PATHS_DICT['AM2K']['TRAIN']['FG_PATH'] and DATASET_PATHS_DICT['AM2K']['TRAIN']['BG_PATH'] in core/config.py
## [2]
## Need to generate the denoised foregrounds for AM-2K and denoised images for BG-20K
## following the paper "Danielyan A, Katkovnik V, Egiazarian K. BM3D frames and variational image deblurring[J]. IEEE Transactions on image processing, 2011"
## and save to path DATASET_PATHS_DICT['AM2K']['TRAIN']['FG_DENOISE_PATH'] and DATASET_PATHS_DICT['BG20K']['TRAIN']['ORIGINAL_DENOISE_PATH'] in core/config.py

batchsizePerGPU=32
GPUNum=1
batchsize=`expr $batchsizePerGPU \* $GPUNum`
backbone='r34'
rosta='TT'
bg_choice='hd'
fg_generate='closed_form'
nEpochs=100
lr=0.00001
threads=8
nickname=gfm_hd_rssn

python core/train.py \
    --logname=$nickname \
    --backbone=$backbone \
    --rosta=$rosta \
    --batchSize=$batchsize \
    --nEpochs=$nEpochs \
    --lr=$lr \
    --threads=$threads \
    --bg_choice=$bg_choice \
    --fg_generate=$fg_generate \
    --model_save_dir=models/trained/$nickname/ \
    --rssn_denoise \