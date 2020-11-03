
#/bin/bash

###########################
## Arch and model_path 
###########################

# arch: e2e_resnet34_2b_gfm_tt | model_path: models/model_r34_2b_gfm_tt.pth

arch='e2e_resnet34_2b_gfm_tt'
test_dataset='SAMPLES'
pred_choice=3

python core/test_samples.py \
     --cuda \
     --arch=$arch \
     --model_path=models/model_r34_2b_gfm_tt.pth \
     --pred_choice=$pred_choice \
     --hybrid \