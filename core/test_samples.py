from config import *
from util import *
import torch
import argparse
import numpy as np
from tqdm import tqdm
from PIL import Image
import cv2
from skimage.transform import resize
from network.e2e_resnet34_2b_gfm_tt import e2e_resnet34_2b_gfm_tt


def get_args():
	# Statement of args
	# --cude: use cude or not
	# --arch: network architectures
	# --model_path: pretrained model path
	# --pred_choice: 1: glance decoder output, 2: focus decoder output, 3: final output
	# --resize: adopt resize testing strategy
	# --hybrid: adopt hybrid testing strategy
	parser = argparse.ArgumentParser(description='PyTorch Super Res Example')
	parser.add_argument('--cuda', action='store_true', help='use cuda?')
	parser.add_argument('--arch', type=str, required=False, default='e2e_resnet34_2b_gfm_tt',choices=["e2e_resnet34_2b_gfm_tt"], help="net backbone")
	parser.add_argument('--model_path', type=str, default='', required=False, help="path of model to use")
	parser.add_argument('--pred_choice', type=int, required=False, default=3, choices=[1, 2, 3], help="pred choice for testing")
	parser.add_argument('--resize', action='store_true', help="resize testing: resize 1/2 for testing")
	parser.add_argument('--hybrid', action='store_true', help="hybrid testing, 1/2 focus + 1/3 glance")
	args = parser.parse_args()
	return args




def inference_function(args, model, scale_img, scale_trimap=None):

	pred_list = []
	tensor_img = torch.from_numpy(scale_img.astype(np.float32)[np.newaxis, :, :, :]).permute(0, 3, 1, 2)
	if args.cuda:
		tensor_img = tensor_img.cuda()
	input_t = tensor_img
	pred_global, pred_local, pred_fusion = model(input_t)

	if args.arch.rfind('tt')>0:
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

	if args.hybrid:
		####
		##Combine 1/3 glance+1/2 focus
		global_ratio = 1/3
		local_ratio = 1/2
		resize_h = int(h*global_ratio)
		resize_w = int(w*global_ratio)
		new_h = min(MAX_SIZE_H, resize_h - (resize_h % 32))
		new_w = min(MAX_SIZE_W, resize_w - (resize_w % 32))
		scale_img = resize(img,(new_h,new_w))*255.0
		pred_glance_1, pred_focus_1, pred_fusion_1 = inference_function(args, model, scale_img)
		pred_glance_1 = resize(pred_glance_1,(h,w))*255.0
		resize_h = int(h*local_ratio)
		resize_w = int(w*local_ratio)
		new_h = min(MAX_SIZE_H, resize_h - (resize_h % 32))
		new_w = min(MAX_SIZE_W, resize_w - (resize_w % 32))
		scale_img = resize(img,(new_h,new_w))*255.0
		pred_glance_2, pred_focus_2, pred_fusion_2 = inference_function(args, model, scale_img)
		pred_focus_2 = resize(pred_focus_2,(h,w))
		if option == 'tt':
			pred_fusion = get_masked_local_from_global_test(pred_glance_1, pred_focus_2)
		elif option == 'bt':
			pred_fusion = pred_glance_1/255.0 - pred_focus_2
			pred_fusion[pred_fusion<0]=0
		else:
			pred_fusion = pred_glance_1/255.0 + pred_focus_2
			pred_fusion[pred_fusion>1]=1
		return [pred_glance_1, pred_focus_2, pred_fusion]

	else:
		if args.resize:
			resize_h = int(h/2)
			resize_w = int(w/2)
			new_h = min(MAX_SIZE_H, resize_h - (resize_h % 32))
			new_w = min(MAX_SIZE_W, resize_w - (resize_w % 32))
		scale_img = resize(img,(new_h,new_w))*255.0
		pred_glance, pred_focus, pred_fusion = inference_function(args, model, scale_img)
		pred_focus = resize(pred_focus,(h,w))
		pred_glance = resize(pred_glance,(h,w))*255.0
		pred_fusion = resize(pred_fusion,(h,w))

		return [pred_glance, pred_focus, pred_fusion]


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
			if args.arch.rfind('tt')>0:
				predict = inference_img_gfm(args, model, img, 'tt')[pred_choice-1]
			elif args.arch.rfind('ft')>0:
				predict = inference_img_gfm(args, model, img, 'ft')[pred_choice-1]
			elif args.arch.rfind('bt')>0:
				predict = inference_img_gfm(args, model, img,'bt')[pred_choice-1]

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
	print(f'Loading arch: {args.arch}')
	print(f'Loading model: {args.model_path}')
	print(f'Predict choice: {args.pred_choice}')
	if args.hybrid:
		print(f'Test strategy: hybrid')
	else:
		print(f'Test strategy: resize')

	if args.arch == 'e2e_resnet34_2b_gfm_tt':
		model = e2e_resnet34_2b_gfm_tt(args)
	ckpt = torch.load(args.model_path)
	model.load_state_dict(ckpt, strict=True)
	if args.cuda:
		model = model.cuda()
	test_samples(args,model)


if __name__ == '__main__':
	args = get_args()
	load_model_and_deploy(args)



