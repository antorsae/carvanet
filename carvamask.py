import argparse
import glob
import numpy as np
import pandas as pd
import random
from scipy.misc import imread, imsave
from os.path import join
from sklearn.model_selection import train_test_split
from skimage.transform import rescale, downscale_local_mean
import scipy.ndimage
import itertools
import re
import os
import sys
import csv
import cv2
from tqdm import tqdm
import jpeg4py as jpeg
from tqdm import tqdm

def rle_decode(mask_rle, shape):
	'''
	mask_rle: run-length as string formated (start length)
	shape: (height,width) of array to return 
	Returns numpy array, 1 - mask, 0 - background

	'''
	s = mask_rle.split()
	starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
	starts -= 1
	ends = starts + lengths
	img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
	for lo, hi in zip(starts, ends):
		img[lo:hi] = 1
	return img.reshape(shape)

parser = argparse.ArgumentParser()
parser.add_argument('-b', '--background-index', action='store_true', help='Build background index')
parser.add_argument('-bb', '--build-background', type=str, help='Build background for filename or all, e.g. -bb fff9b3a5373f_04.jpg or -bb all')
parser.add_argument('-i', '--idx', type=int, nargs='+', help='Indexes to use, e.g. -i 2 16')
parser.add_argument('-cm', '--coarse-mask', type=str, help='Use coarse mask for backgroung generation, e.g. -cm train_masks.csv')
parser.add_argument('-t', '--test', action='store_true', help='Generate test backgrounds (instead of train) ')

args = parser.parse_args()

if args.idx:
	IMGS_IDX = args.idx
else:
	IMGS_IDX = range(1,17)

IDXS     = len(IMGS_IDX)

SX   = 1918
SY   = 1280
if args.test:
	TRAIN_FOLDER 	  = 'test_hq'
	BACKGROUND_FOLDER = 'test_background_hq'
else:
	TRAIN_FOLDER 	  = 'train_hq'
	BACKGROUND_FOLDER = 'train_background_hq'

BACKORDER_FILENAME = 'background_order.npy'

ids = list(set([(x.split('/')[1]).split('_')[0] for x in glob.glob(join(TRAIN_FOLDER, '*_*.jpg'))]))
ids.sort()

load_mask  = lambda im, idx: imread(join('train_masks', '{}_{:02d}_mask.gif'.format(im, idx))).astype('uint8') # / 255. ).astype('float32') #/ 255.
load_img   = lambda im, idx: jpeg.JPEG(join(TRAIN_FOLDER, '{}_{:02d}.jpg'.format(im, idx))).decode()[:SY, ...].astype(np.float32) / 255.

if args.background_index:
	assert args.test is False
	background_counts = np.zeros((SY,SX,IDXS), dtype=np.int32)
	for item in tqdm(ids):
		for idx in IMGS_IDX:
			mask = load_mask(item, idx)[...,0] / 255
			background_counts[...,idx-1] += (1 - mask)
	max0 = np.amax(background_counts[...,0])
	#imsave("back0.png", background_counts[...,0].astype(np.float32) / max0)
	background_order = np.argsort(background_counts, axis=2)
	for i in range(IDXS):
		max0 = np.amax(background_counts[...,i])
		for j in range(IDXS):
			background_order[(background_order[...,j] == i) & (background_counts[...,i] < max0 * 0.9), j] = -1 
	#imsave("backo0.png", background_order[...,IDXS-1].astype(np.float32) / (IDXS-1))
	np.save(BACKORDER_FILENAME, background_order)

elif args.build_background:

	if args.build_background != 'all':
		_this_car, _this_idx = args.build_background.split('.')[0].split('_')
		_this_idx = [int(_this_idx)]
		_this_car = [_this_car]
	else:
		_this_car = ids
		_this_idx = IMGS_IDX

	car_views = np.empty((SY, SX, 3, IDXS+1), dtype=np.float32)
	car_views[...,0] = (1.,0.,1.)
	background_img = np.empty((SY,SX,3), dtype=np.float32)

	if not args.coarse_mask:
		background_order = np.load(BACKORDER_FILENAME)
		background_order += 1
		BACK_INDEXES = 1
		background_index = np.empty((SY,SX,BACK_INDEXES), dtype=np.int32)
	else:
		with open(args.coarse_mask, 'rb') as csvfile:
			reader = csv.reader(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
			rle_dict = {rows[0]: rows[1] for rows in reader}
		dilation_kernel = np.ones((5,5),np.uint8)
		car_masks = np.empty((SY, SX, IDXS), dtype=np.uint8)

	for this_car in tqdm(_this_car):
		cars_backgrounds = np.empty((SY,SX,3,IDXS), dtype=np.float32)

		for idx in IMGS_IDX:
			car_views[...,idx] = load_img(this_car, idx)

			if args.coarse_mask:
				fname = '{}_{:02d}.jpg'.format(this_car, idx)
				assert fname in rle_dict
				car_masks[...,idx-1] = cv2.dilate(rle_decode(rle_dict[fname], (1280, 1918) ),dilation_kernel,iterations = 1)
				cars_backgrounds[...,idx-1] = np.copy(car_views[...,idx])
				cars_backgrounds[car_masks[...,idx-1] == 1,:, idx-1] = np.nan

		if args.coarse_mask:

			def reject_outliers(data, m = 5.,mode='median'):
				if mode == 'median':
					d = np.abs(data - np.expand_dims(np.nanmedian(data, axis=3), axis=3))
				elif mode == 'min':
					d = np.abs(data - np.expand_dims(np.nanmin(data, axis=3), axis=3))
				mdev = np.expand_dims(np.nanmedian(d, axis=3), axis=3)
				s = d/mdev
				data[s > m ] = np.nan
				return data

			car_mask_ovelaid = np.sum(car_masks, axis=2)
			background_candidates = car_mask_ovelaid < 16-2
			cars_backgrounds[:1280//2,...] = reject_outliers(cars_backgrounds[:1280//2,...])
			cars_backgrounds[1280//2:,...] = reject_outliers(cars_backgrounds[1280//2:,...], mode='min')

			background_img = np.nanmedian(cars_backgrounds, axis=3)

			background_img[~background_candidates,:] = (1.,0.,1.)
			imsave(join(BACKGROUND_FOLDER,'{}.png'.format(this_car)), background_img)

		else:
			for back in range(BACK_INDEXES):
				background_index[...,back] = background_order[...,IDXS-1-back]

			# TODO: do smarter mean if BACK_INDEXES != 1
			background_index = background_index[...,0].reshape((SY,SX,1))

			background_index = np.repeat(background_index, repeats = 3, axis=2)
			for i in range(IDXS+1):
				background_img = np.select([background_index == i, background_index != i], [car_views[...,i], background_img])

			imsave(join(BACKGROUND_FOLDER,'{}.png'.format(this_car)), background_img)

else:

	# PLAYGROUND AREA. Have fun!
	areas = []
	for item in ids:
		for idx in IMGS_IDX:
			print(item, idx)
			mask = load_mask(item, idx)[...,0]
			ret,thresh = cv2.threshold(mask,127,255,cv2.THRESH_BINARY)

			#print(thresh.dtype, thresh.shape, np.amax(thresh), np.amin(thresh))
			_, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
			for cnt in contours:
				area = cv2.contourArea(cnt)
				#print(area)
				areas.append([area, item, idx])

	h = np.histogram([area[0] for area in areas], bins=50)
	print(h)



