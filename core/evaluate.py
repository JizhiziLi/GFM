"""
Bridging Composite and Real: Towards End-to-end Deep Image Matting [IJCV-2021]
Evaluation files for training and testing.

Copyright (c) 2021, Jizhizi Li (jili8515@uni.sydney.edu.au)
Licensed under the MIT License (see LICENSE for details)
Github repo: https://github.com/JizhiziLi/GFM
Paper link (Arxiv): https://arxiv.org/abs/2010.16188

"""

import numpy as np
import torch
import torch.nn as nn
import math
from torch.autograd import Variable
import torch.nn.functional as fnn

##########################
### Training loses for GFM
##########################
def get_crossentropy_loss(output_class, gt,pre):	
	gt_copy = gt.clone()
	if output_class==2:
		gt_copy[gt_copy<255]=0
		gt_copy[gt_copy>0]=1
	elif output_class==3:
		gt_copy[gt_copy==0] = 0
		gt_copy[gt_copy==255] = 2
		gt_copy[gt_copy>2] = 1
	gt_copy = gt_copy.long()
	gt_copy = gt_copy[:,0,:,:]
	criterion = nn.CrossEntropyLoss()
	entropy_loss = criterion(pre, gt_copy)
	return entropy_loss

def get_alpha_loss(predict, alpha, trimap):
	weighted = torch.zeros(trimap.shape).cuda()
	weighted[trimap == 128] = 1.
	alpha_f = alpha / 255.
	alpha_f = alpha_f.cuda()
	diff = predict - alpha_f
	diff = diff * weighted
	alpha_loss = torch.sqrt(diff ** 2 + 1e-12)
	alpha_loss_weighted = alpha_loss.sum() / (weighted.sum() + 1.)
	return alpha_loss_weighted

def get_alpha_loss_whole_img(predict, alpha):
	weighted = torch.ones(alpha.shape).cuda()
	alpha_f = alpha / 255.
	alpha_f = alpha_f.cuda()
	diff = predict - alpha_f
	alpha_loss = torch.sqrt(diff ** 2 + 1e-12)
	alpha_loss = alpha_loss.sum()/(weighted.sum())
	return alpha_loss

## Laplacian loss is refer to 
## https://gist.github.com/MarcoForte/a07c40a2b721739bb5c5987671aa5270
def build_gauss_kernel(size=5, sigma=1.0, n_channels=1, cuda=False):
	if size % 2 != 1:
		raise ValueError("kernel size must be uneven")
	grid = np.float32(np.mgrid[0:size,0:size].T)
	gaussian = lambda x: np.exp((x - size//2)**2/(-2*sigma**2))**2
	kernel = np.sum(gaussian(grid), axis=2)
	kernel /= np.sum(kernel)
	kernel = np.tile(kernel, (n_channels, 1, 1))
	kernel = torch.FloatTensor(kernel[:, None, :, :]).cuda()
	return Variable(kernel, requires_grad=False)

def conv_gauss(img, kernel):
	""" convolve img with a gaussian kernel that has been built with build_gauss_kernel """
	n_channels, _, kw, kh = kernel.shape
	img = fnn.pad(img, (kw//2, kh//2, kw//2, kh//2), mode='replicate')
	return fnn.conv2d(img, kernel, groups=n_channels)

def laplacian_pyramid(img, kernel, max_levels=5):
	current = img
	pyr = []
	for level in range(max_levels):
		filtered = conv_gauss(current, kernel)
		diff = current - filtered
		pyr.append(diff)
		current = fnn.avg_pool2d(filtered, 2)
	pyr.append(current)
	return pyr

def get_laplacian_loss(predict, alpha, trimap):
	weighted = torch.zeros(trimap.shape).cuda()
	weighted[trimap == 128] = 1.
	alpha_f = alpha / 255.
	alpha_f = alpha_f.cuda()
	alpha_f = alpha_f.clone()*weighted
	predict = predict.clone()*weighted
	gauss_kernel = build_gauss_kernel(size=5, sigma=1.0, n_channels=1, cuda=True)
	pyr_alpha  = laplacian_pyramid(alpha_f, gauss_kernel, 5)
	pyr_predict = laplacian_pyramid(predict, gauss_kernel, 5)
	laplacian_loss_weighted = sum(fnn.l1_loss(a, b) for a, b in zip(pyr_alpha, pyr_predict))
	return laplacian_loss_weighted

def get_laplacian_loss_whole_img(predict, alpha):
	alpha_f = alpha / 255.
	alpha_f = alpha_f.cuda()
	gauss_kernel = build_gauss_kernel(size=5, sigma=1.0, n_channels=1, cuda=True)
	pyr_alpha  = laplacian_pyramid(alpha_f, gauss_kernel, 5)
	pyr_predict = laplacian_pyramid(predict, gauss_kernel, 5)
	laplacian_loss = sum(fnn.l1_loss(a, b) for a, b in zip(pyr_alpha, pyr_predict))
	return laplacian_loss

def get_composition_loss_whole_img(img, alpha, fg, bg, predict):
	weighted = torch.ones(alpha.shape).cuda()
	predict_3 = torch.cat((predict, predict, predict), 1)
	comp = predict_3 * fg + (1. - predict_3) * bg
	comp_loss = torch.sqrt((comp - img) ** 2 + 1e-12) / 255.
	comp_loss = comp_loss.sum()/(weighted.sum())
	return comp_loss


##########################
### Testing loses for GFM
##########################
def calculate_sad_mse_mad(predict_old,alpha,trimap):
	predict = np.copy(predict_old)
	pixel = float((trimap == 128).sum())
	predict[trimap == 255] = 1.
	predict[trimap == 0  ] = 0.
	sad_diff = np.sum(np.abs(predict - alpha))/1000
	if pixel==0:
		pixel = trimap.shape[0]*trimap.shape[1]-float((trimap==255).sum())-float((trimap==0).sum())
	mse_diff = np.sum((predict - alpha) ** 2)/pixel
	mad_diff = np.sum(np.abs(predict - alpha))/pixel
	return sad_diff, mse_diff, mad_diff
	
def calculate_sad_mse_mad_whole_img(predict, alpha):
	pixel = predict.shape[0]*predict.shape[1]
	sad_diff = np.sum(np.abs(predict - alpha))/1000
	mse_diff = np.sum((predict - alpha) ** 2)/pixel
	mad_diff = np.sum(np.abs(predict - alpha))/pixel
	return sad_diff, mse_diff, mad_diff	

def calculate_sad_fgbg(predict, alpha, trimap):
	sad_diff = np.abs(predict-alpha)
	weight_fg = np.zeros(predict.shape)
	weight_bg = np.zeros(predict.shape)
	weight_trimap = np.zeros(predict.shape)
	weight_fg[trimap==255] = 1.
	weight_bg[trimap==0  ] = 1.
	weight_trimap[trimap==128  ] = 1.
	sad_fg = np.sum(sad_diff*weight_fg)/1000
	sad_bg = np.sum(sad_diff*weight_bg)/1000
	sad_trimap = np.sum(sad_diff*weight_trimap)/1000
	return sad_fg, sad_bg

def compute_gradient_whole_image(pd, gt):
	from scipy.ndimage import gaussian_filter

	pd_x = gaussian_filter(pd, sigma=1.4, order=[1, 0], output=np.float32)
	pd_y = gaussian_filter(pd, sigma=1.4, order=[0, 1], output=np.float32)
	gt_x = gaussian_filter(gt, sigma=1.4, order=[1, 0], output=np.float32)
	gt_y = gaussian_filter(gt, sigma=1.4, order=[0, 1], output=np.float32)
	pd_mag = np.sqrt(pd_x**2 + pd_y**2)
	gt_mag = np.sqrt(gt_x**2 + gt_y**2)

	error_map = np.square(pd_mag - gt_mag)
	loss = np.sum(error_map) / 10
	return loss

def compute_connectivity_loss_whole_image(pd, gt, step=0.1):
	from scipy.ndimage import morphology
	from skimage.measure import label, regionprops
	h, w = pd.shape
	thresh_steps = np.arange(0, 1.1, step)
	l_map = -1 * np.ones((h, w), dtype=np.float32)
	lambda_map = np.ones((h, w), dtype=np.float32)
	for i in range(1, thresh_steps.size):
		pd_th = pd >= thresh_steps[i]
		gt_th = gt >= thresh_steps[i]
		label_image = label(pd_th & gt_th, connectivity=1)
		cc = regionprops(label_image)
		size_vec = np.array([c.area for c in cc])
		if len(size_vec) == 0:
			continue
		max_id = np.argmax(size_vec)
		coords = cc[max_id].coords
		omega = np.zeros((h, w), dtype=np.float32)
		omega[coords[:, 0], coords[:, 1]] = 1
		flag = (l_map == -1) & (omega == 0)
		l_map[flag == 1] = thresh_steps[i-1]
		dist_maps = morphology.distance_transform_edt(omega==0)
		dist_maps = dist_maps / dist_maps.max()
	l_map[l_map == -1] = 1
	d_pd = pd - l_map
	d_gt = gt - l_map
	phi_pd = 1 - d_pd * (d_pd >= 0.15).astype(np.float32)
	phi_gt = 1 - d_gt * (d_gt >= 0.15).astype(np.float32)
	loss = np.sum(np.abs(phi_pd - phi_gt)) / 1000
	return loss