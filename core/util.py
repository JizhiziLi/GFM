import os
import shutil
import torch
import numpy as np
import cv2


##########################
### Pure functions
##########################
def extract_pure_name(original_name):
	pure_name, extention = os.path.splitext(original_name)
	return pure_name

def listdir_nohidden(path):
	new_list = []
	for f in os.listdir(path):
		if not f.startswith('.'):
			new_list.append(f)
	new_list.sort()
	return new_list

def create_folder_if_not_exist(folder_path):
	if not os.path.exists(folder_path):
		os.makedirs(folder_path)

def refresh_folder(folder_path):
	if not os.path.exists(folder_path):
		os.makedirs(folder_path)
	else:
		shutil.rmtree(folder_path)
		os.makedirs(folder_path)

def generate_composite_img(img, alpha_channel):
	b_channel, g_channel, r_channel = cv2.split(img)
	b_channel = b_channel * alpha_channel
	g_channel = g_channel * alpha_channel
	r_channel = r_channel * alpha_channel
	alpha_channel = (alpha_channel*255).astype(b_channel.dtype)
	img_BGRA = cv2.merge((r_channel,g_channel,b_channel,alpha_channel))
	return img_BGRA


##########################
### Functions for fusion 
##########################
def get_masked_local_from_global(glance_sigmoid, focus_sigmoid):
	values, index = torch.max(glance_sigmoid,1)
	index = index[:,None,:,:].float()
	### index <===> [0, 1, 2]
	### bg_mask <===> [1, 0, 0]
	bg_mask = index.clone()
	bg_mask[bg_mask==2]=1
	bg_mask = 1- bg_mask
	### trimap_mask <===> [0, 1, 0]
	trimap_mask = index.clone()
	trimap_mask[trimap_mask==2]=0
	### fg_mask <===> [0, 0, 1]
	fg_mask = index.clone()
	fg_mask[fg_mask==1]=0
	fg_mask[fg_mask==2]=1
	focus_sigmoid = focus_sigmoid.cpu()
	trimap_mask = trimap_mask.cpu()
	fg_mask = fg_mask.cpu()
	fusion_sigmoid = focus_sigmoid*trimap_mask+fg_mask
	return fusion_sigmoid

def get_masked_local_from_global_test(global_result, local_result):
	weighted_global = np.ones(global_result.shape)
	weighted_global[global_result==255] = 0
	weighted_global[global_result==0] = 0
	fusion_result = global_result*(1.-weighted_global)/255+local_result*weighted_global
	return fusion_result
	
def gen_trimap_from_segmap_e2e(segmap):

	trimap = np.argmax(segmap, axis=1)[0]
	trimap = trimap.astype(np.int64)	
	trimap[trimap==1]=128
	trimap[trimap==2]=255
	return trimap.astype(np.uint8)
