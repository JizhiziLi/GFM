#/bin/bash
## An easier way to train GFM on ORI-Track
## No need for generating foreground and backgrounds in advance

batchsizePerGPU=32
GPUNum=1
batchsize=`expr $batchsizePerGPU \* $GPUNum`
backbone='r34'
rosta='TT'
bg_choice='original'
fg_generate='alpha_blending'
nEpochs=500
lr=0.00001
threads=8
nickname=gfm_ori_track_easier

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