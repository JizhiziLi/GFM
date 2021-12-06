"""
Bridging Composite and Real: Towards End-to-end Deep Image Matting [IJCV-2021]
Main test file.

Copyright (c) 2021, Jizhizi Li (jili8515@uni.sydney.edu.au)
Licensed under the MIT License (see LICENSE for details)
Github repo: https://github.com/JizhiziLi/GFM
Paper link (Arxiv): https://arxiv.org/abs/2010.16188

"""

import torch
import argparse
import numpy as np
from tqdm import tqdm
from PIL import Image
import cv2
from skimage.transform import resize
import logging
from config import *
from util import *
from evaluate import *
from gfm import GFM


def get_args():
	
	parser = argparse.ArgumentParser(description='Arguments for the testing purpose.')	
	# backbone: the backbone of GFM, we provide four backbones - r34, r34_2b, d121 and r101.
	# rosta (Representations of Semantic and Transition areas): we provide three types - TT, FT, and BT. 
	# We also present RIM indicates RoSTa Integration Module.
	# model_path: path of the pretrained model to use
	# pred_choice(TT/FT/BT): 1 (glance decoder), 2 (focus decoder) and 3 (final result after Collaborative Matting) 
	# pred_choice(RIM): 1 (TT result), 2 (FT result), 3 (BT result), and 4 (RIM result). 
	# test_choice: test strategy (HYBRID or RESIZE)
	# test_result_dir: path to save the test results
	# logname: name of the logging files
	parser.add_argument('--cuda', action='store_true', help='use cuda?')
	parser.add_argument('--backbone', type=str, required=False, default='r34',choices=["r34","r34_2b","d121","r101"], help="net backbone")
	parser.add_argument('--rosta', type=str, required=False, default='TT',choices=["TT","FT","BT","RIM"], help="rosta")
	parser.add_argument('--model_path', type=str, default='', required=False, help="path of the pretrained model to use")
	parser.add_argument('--pred_choice', type=int, required=False, default=3, choices=[1, 2, 3, 4], help="pred choice for testing, three options for TT/FT/BT, four for RIM")
	parser.add_argument('--dataset_choice', type=str, required=True, choices=['AM_2K','SAMPLES'], help="which dataset to test")
	parser.add_argument('--test_choice', type=str, required=True, choices=['RESIZE','HYBRID'], help="which test strategy to use")
	parser.add_argument('--test_result_dir', type=str, required=False, default='', help="where to save the test results")
	parser.add_argument('--logname', type=str, default='test_log', required=False, help="name of the logging file")
	args = parser.parse_args()
	return args


def inference_img_scale(args, model, scale_img):

	pred_list = []
	tensor_img = torch.from_numpy(scale_img.astype(np.float32)[np.newaxis, :, :, :]).permute(0, 3, 1, 2)
	if args.cuda:
		tensor_img = tensor_img.cuda()	
	input_t = tensor_img
	if args.rosta=='RIM':
		pred_tt, pred_ft, pred_bt, pred_fusion = model(input_t)
		pred_tt = pred_tt[2].data.cpu().numpy()[0,0,:,:]
		pred_ft = pred_ft[2].data.cpu().numpy()[0,0,:,:]
		pred_bt = pred_bt[2].data.cpu().numpy()[0,0,:,:]
		pred_fusion = pred_fusion.data.cpu().numpy()[0,0,:,:]
		return pred_tt, pred_ft, pred_bt, pred_fusion
	else:
		pred_global, pred_local, pred_fusion = model(input_t)
		if args.rosta == 'TT':
			pred_global = pred_global.data.cpu().numpy()
			pred_global = gen_trimap_from_segmap_e2e(pred_global)
		else:
			pred_global = pred_global.data.cpu().numpy()
			pred_global = gen_bw_from_segmap_e2e(pred_global)
		pred_local = pred_local.data.cpu().numpy()[0,0,:,:]
		pred_fusion = pred_fusion.data.cpu().numpy()[0,0,:,:]

		return pred_global, pred_local, pred_fusion


def inference_img_gfm(args, model, img, option):

	h, w, c = img.shape
	new_h = min(MAX_SIZE_H, h - (h % 32))
	new_w = min(MAX_SIZE_W, w - (w % 32))

	if args.rosta=='RIM':
		resize_h = int(h/3)
		resize_w = int(w/3)
		new_h = min(MAX_SIZE_H, resize_h - (resize_h % 32))
		new_w = min(MAX_SIZE_W, resize_w - (resize_w % 32))
		scale_img = resize(img,(new_h,new_w))*255.0
		pred_tt, pred_ft, pred_bt, pred_fusion = inference_img_scale(args, model, scale_img)
		pred_tt = resize(pred_tt,(h,w))
		pred_ft = resize(pred_ft,(h,w))
		pred_bt = resize(pred_bt,(h,w))
		pred_fusion = resize(pred_fusion,(h,w))
		return [pred_tt, pred_ft, pred_bt, pred_fusion]
	elif args.test_choice=='RESIZE':
		resize_h = int(h/2)
		resize_w = int(w/2)
		new_h = min(MAX_SIZE_H, resize_h - (resize_h % 32))
		new_w = min(MAX_SIZE_W, resize_w - (resize_w % 32))
		scale_img = resize(img,(new_h,new_w))*255.0
		pred_glance, pred_focus, pred_fusion = inference_img_scale(args, model, scale_img)
		pred_focus = resize(pred_focus,(h,w))
		pred_glance = resize(pred_glance,(h,w))*255.0
		pred_fusion = resize(pred_fusion,(h,w))
		return [pred_glance, pred_focus, pred_fusion]
	else:
		## Combine 1/3 glance and 1/2 focus
		global_ratio = 1/3
		local_ratio = 1/2
		resize_h = int(h*global_ratio)
		resize_w = int(w*global_ratio)
		new_h = min(MAX_SIZE_H, resize_h - (resize_h % 32))
		new_w = min(MAX_SIZE_W, resize_w - (resize_w % 32))
		scale_img = resize(img,(new_h,new_w))*255.0
		pred_glance_1, pred_focus_1, pred_fusion_1 = inference_img_scale(args, model, scale_img)
		pred_glance_1 = resize(pred_glance_1,(h,w))*255.0
		resize_h = int(h*local_ratio)
		resize_w = int(w*local_ratio)
		new_h = min(MAX_SIZE_H, resize_h - (resize_h % 32))
		new_w = min(MAX_SIZE_W, resize_w - (resize_w % 32))
		scale_img = resize(img,(new_h,new_w))*255.0
		pred_glance_2, pred_focus_2, pred_fusion_2 = inference_img_scale(args, model, scale_img)
		pred_focus_2 = resize(pred_focus_2,(h,w))
		if option == 'TT':
			pred_fusion = get_masked_local_from_global_test(pred_glance_1, pred_focus_2)
		elif option == 'BT':
			pred_fusion = pred_glance_1/255.0 - pred_focus_2
			pred_fusion[pred_fusion<0]=0
		else:
			pred_fusion = pred_glance_1/255.0 + pred_focus_2
			pred_fusion[pred_fusion>1]=1
		return [pred_glance_1, pred_focus_2, pred_fusion]

		

def test_am2k(args, model):
	############################
	# Some initial setting for paths
	############################
	ORIGINAL_PATH = DATASET_PATHS_DICT['AM2K']['VALIDATION']['ORIGINAL_PATH']
	MASK_PATH = DATASET_PATHS_DICT['AM2K']['VALIDATION']['MASK_PATH']
	TRIMAP_PATH = DATASET_PATHS_DICT['AM2K']['VALIDATION']['TRIMAP_PATH']
	pred_choice = args.pred_choice

	############################
	# Start testing
	############################
	sad_diffs = 0.
	mse_diffs = 0.
	mad_diffs = 0.
	grad_diffs = 0.
	conn_diffs = 0.
	sad_trimap_diffs = 0.
	mse_trimap_diffs = 0.
	mad_trimap_diffs = 0.
	sad_fg_diffs = 0.
	sad_bg_diffs = 0.

	refresh_folder(args.test_result_dir)
	model.eval()
	img_list = listdir_nohidden(ORIGINAL_PATH)

	total_number = len(img_list)
	args.logging.info("===============================")
	args.logging.info(f'====> Start Testing\n\t--Dataset: {args.dataset_choice}\n\t--Test: {args.test_choice}\n\t--Number: {total_number}')

	for img_name in tqdm(img_list):

		img_path = ORIGINAL_PATH+img_name
		alpha_path = MASK_PATH+extract_pure_name(img_name)+'.png'
		trimap_path = TRIMAP_PATH+extract_pure_name(img_name)+'.png'
		img = np.array(Image.open(img_path))
		trimap = np.array(Image.open(trimap_path))
		alpha = np.array(Image.open(alpha_path))/255.
		img = img[:,:,:3] if img.ndim>2 else img
		trimap = trimap[:,:,0] if trimap.ndim>2 else trimap
		alpha = alpha[:,:,0] if alpha.ndim>2 else alpha

		with torch.no_grad():
			if args.cuda:
				torch.cuda.empty_cache()
			predict = inference_img_gfm(args, model, img, args.rosta)[pred_choice-1]

			if pred_choice==1:
				cv2.imwrite(os.path.join(args.test_result_dir, extract_pure_name(img_name)+'.png'),predict.astype(np.uint8))
			elif pred_choice==2:
				save_test_result(os.path.join(args.test_result_dir, extract_pure_name(img_name)+'.png'),predict)
			else:
				sad_trimap_diff, mse_trimap_diff, mad_trimap_diff = calculate_sad_mse_mad(predict, alpha, trimap)
				sad_diff, mse_diff, mad_diff = calculate_sad_mse_mad_whole_img(predict, alpha)
				sad_fg_diff, sad_bg_diff = calculate_sad_fgbg(predict, alpha, trimap)
				conn_diff = compute_connectivity_loss_whole_image(predict, alpha)
				grad_diff = compute_gradient_whole_image(predict, alpha)

				args.logging.info(f"[{img_list.index(img_name)+1}/{total_number}]\nImage:{img_name}\nsad:{sad_diff}\nmse:{mse_diff}\nmad:{mad_diff}\nsad_trimap:{sad_trimap_diff}\nmse_trimap:{mse_trimap_diff}\nmad_trimap:{mad_trimap_diff}\nsad_fg:{sad_fg_diff}\nsad_bg:{sad_bg_diff}\nconn:{conn_diff}\ngrad:{grad_diff}\n-----------")
				sad_diffs += sad_diff
				mse_diffs += mse_diff
				mad_diffs += mad_diff
				mse_trimap_diffs += mse_trimap_diff
				sad_trimap_diffs += sad_trimap_diff
				mad_trimap_diffs += mad_trimap_diff
				sad_fg_diffs += sad_fg_diff
				sad_bg_diffs += sad_bg_diff
				conn_diffs += conn_diff
				grad_diffs += grad_diff
				save_test_result(os.path.join(args.test_result_dir, extract_pure_name(img_name)+'.png'),predict)

	args.logging.info("===============================")
	args.logging.info(f"Testing numbers: {total_number}")

	if pred_choice in [3,4]:
		args.logging.info("SAD: {}".format(sad_diffs / total_number))
		args.logging.info("MSE: {}".format(mse_diffs / total_number))
		args.logging.info("MAD: {}".format(mad_diffs / total_number))
		args.logging.info("GRAD: {}".format(grad_diffs / total_number))
		args.logging.info("CONN: {}".format(conn_diffs / total_number))
		args.logging.info("SAD TRIMAP: {}".format(sad_trimap_diffs / total_number))
		args.logging.info("MSE TRIMAP: {}".format(mse_trimap_diffs / total_number))
		args.logging.info("MAD TRIMAP: {}".format(mad_trimap_diffs / total_number))
		args.logging.info("SAD FG: {}".format(sad_fg_diffs / total_number))
		args.logging.info("SAD BG: {}".format(sad_bg_diffs / total_number))
		return int(sad_diffs/total_number)
	else:
		return 0


def test_samples(args, model):

	print(f'=====> Test on samples and save alpha, color results')
	model.eval()
	pred_choice = args.pred_choice

	img_list = listdir_nohidden(SAMPLES_ORIGINAL_PATH)
	refresh_folder(SAMPLES_RESULT_ALPHA_PATH)
	if pred_choice==3:
		refresh_folder(SAMPLES_RESULT_COLOR_PATH)

	for img_name in tqdm(img_list):
		img_path = SAMPLES_ORIGINAL_PATH+img_name
		try:
			img = np.array(Image.open(img_path))[:,:,:3]
		except Exception as e:
			print(f'Error: {str(e)} | Name: {img_name}')
		h, w, c = img.shape
		if min(h, w)>SHORTER_PATH_LIMITATION:
		  if h>=w:
			  new_w = SHORTER_PATH_LIMITATION
			  new_h = int(SHORTER_PATH_LIMITATION*h/w)
			  img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
		  else:
			  new_h = SHORTER_PATH_LIMITATION
			  new_w = int(SHORTER_PATH_LIMITATION*w/h)
			  img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

		with torch.no_grad():
			if args.cuda:
				torch.cuda.empty_cache()

			predict = inference_img_gfm(args, model, img, args.rosta)[pred_choice-1]
			
		if pred_choice==3:
			composite = generate_composite_img(img, predict)
			cv2.imwrite(os.path.join(SAMPLES_RESULT_COLOR_PATH, extract_pure_name(img_name)+'.png'),composite)
			predict = predict*255.0
			cv2.imwrite(os.path.join(SAMPLES_RESULT_ALPHA_PATH, extract_pure_name(img_name)+'.png'),predict.astype(np.uint8))
		if pred_choice==2:
			predict = predict*255.0
			cv2.imwrite(os.path.join(SAMPLES_RESULT_ALPHA_PATH, extract_pure_name(img_name)+'.png'),predict.astype(np.uint8))
		else:
			cv2.imwrite(os.path.join(SAMPLES_RESULT_ALPHA_PATH, extract_pure_name(img_name)+'.png'),predict.astype(np.uint8))


def load_model_and_deploy(args):

	print('*********************************')
	print(f'Loading backbone: {args.backbone}')
	print(f'Loading rosta: {args.rosta}')
	print(f'Loading model: {args.model_path}')
	print(f'Predict choice: {args.pred_choice}')
	print(f'Test strategy: {args.test_choice}')
	print(f'Saving to the folder: {args.test_result_dir}')

	model = GFM(args)

	if torch.cuda.device_count()==0:
		print(f'Running on CPU...')
		args.cuda = False
		ckpt = torch.load(args.model_path,map_location=torch.device('cpu'))
	else:
		print(f'Running on GPU with CUDA as {args.cuda}...')
		ckpt = torch.load(args.model_path)

	model.load_state_dict(ckpt, strict=True)
	if args.cuda:
		model = model.cuda()

	if args.dataset_choice=='SAMPLES':
		test_samples(args,model)
	elif args.dataset_choice=='AM_2K':
		logging_filename = TEST_LOGS_FOLDER+args.logname+'.log'
		if os.path.exists(logging_filename):
			os.remove(logging_filename)
		logging.basicConfig(filename=logging_filename, level=logging.INFO)
		args.logging = logging
		test_am2k(args, model)


if __name__ == '__main__':
	args = get_args()
	load_model_and_deploy(args)



