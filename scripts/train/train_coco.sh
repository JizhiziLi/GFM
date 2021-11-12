#/bin/bash

batchsizePerGPU=32
GPUNum=1
batchsize=`expr $batchsizePerGPU \* $GPUNum`
backbone='r34'
rosta='TT'
bg_choice='coco'
fg_generate='closed_form'
nEpochs=100
lr=0.00001
threads=8
nickname=gfm_coco

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