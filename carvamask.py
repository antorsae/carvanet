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

parser = argparse.ArgumentParser()
parser.add_argument('-b', '--background-index', action='store_true', help='Build background index')
parser.add_argument('-bb', '--build-background', type=str, help='Build background for filename (e.g. -bb fff9b3a5373f_04.jpg')
parser.add_argument('-i', '--idx', type=int, nargs='+', help='Indexes to use, e.g. -i 2 16')

args = parser.parse_args()

if args.idx:
    IMGS_IDX = args.idx
else:
    IMGS_IDX = range(1,17)

IDXS     = len(IMGS_IDX)

SX   = 1918
SY   = 1280

TRAIN_FOLDER = 'train_hq'
BACKGROUND_FOLDER = 'background_hq'
BACKORDER_FILENAME = 'background_order.npy'

ids = list(set([(x.split('/')[1]).split('_')[0] for x in glob.glob(join(TRAIN_FOLDER, '*_*.jpg'))]))
ids.sort()
#ids_train = list(itertools.product(ids_train, IMGS_IDX))

load_mask  = lambda im, idx: imread(join('train_masks', '{}_{:02d}_mask.gif'.format(im, idx))).astype('uint8') # / 255. ).astype('float32') #/ 255.
load_img   = lambda im, idx: jpeg.JPEG(join(TRAIN_FOLDER, '{}_{:02d}.jpg'.format(im, idx))).decode()[:SY, ...].astype(np.float32) / 255.

if args.background_index:
	background_counts = np.zeros((SY,SX,IDXS), dtype=np.int32)
	for item in tqdm(ids):
		for idx in IMGS_IDX:
			mask = load_mask(item, idx)[...,0] / 255
			background_counts[...,idx-1] += (1 - mask)
	max0 = np.amax(background_counts[...,0])
	imsave("back0.png", background_counts[...,0].astype(np.float32) / max0)
	background_order = np.argsort(background_counts, axis=2)
	imsave("backo0.png", background_order[...,0].astype(np.float32) / (IDXS-1))
	np.save(BACKORDER_FILENAME, background_order)

elif args.build_background:

	background_order = np.load(BACKORDER_FILENAME)

	if args.build_background != 'all':
		_this_car, _this_idx = args.build_background.split('.')[0].split('_')
		_this_idx = [int(_this_idx)]
		_this_car = [_this_car]
	else:
		_this_car = ids
		_this_idx = IMGS_IDX

	BACK_INDEXES = 1

	car_views = np.empty((SY, SX, 3, IDXS), dtype=np.float32)
	background_index = np.empty((SY,SX,BACK_INDEXES), dtype=np.int32)
	background_img = np.empty((SY,SX,3), dtype=np.float32)

	for this_car in tqdm(_this_car):
		for idx in IMGS_IDX:
			car_views[...,idx-1] = load_img(this_car, idx)

		for this_idx in _this_idx:
			for back in range(BACK_INDEXES):
				background_index[...,back] = background_order[...,IDXS-1-back]
				background_index[background_index[...,back] == this_idx, back] = background_order[background_index[...,back] == this_idx, IDXS-2-back]

			background_index = background_index[...,0].reshape((SY,SX,1))
			assert np.all(background_index != this_idx)

			background_index = np.repeat(background_index, repeats = 3, axis=2)
			for i in range(IDXS):
				background_img = np.select([background_index == i, background_index != i], [car_views[...,i], background_img])

			imsave(join(BACKGROUND_FOLDER,'{}_{:02d}.jpg'.format(this_car, this_idx)), background_img)

else:

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



