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

parser = argparse.ArgumentParser()
parser.add_argument('--max-epoch', type=int, default=200, help='Epoch to run')
parser.add_argument('-b', '--batch-size', type=int, default=1, help='Batch Size during training, e.g. -b 2')
parser.add_argument('-ba', '--batch-acc', type=int, default=1, help='Batch Size for training accumulation, e.g. -b 4')
parser.add_argument('-l', '--learning_rate', type=float, default=1e-3, help='Initial learning rate')
parser.add_argument('-m', '--model', help='load hdf5 model (and continue training)')
parser.add_argument('-s', '--scale', type=int, default=1, help='downscale e.g. -s 2')
parser.add_argument('-ra', '--rotation', type=float, default=0.5, help='Rotation angle for augmentation e.g. -r 2')
parser.add_argument('-u', '--unet', action='store_true', help='use UNET')
parser.add_argument('-r', '--resnet', action='store_true', help='use residual dilated nets')
parser.add_argument('-yuv', '--yuv', action='store_true', help='Use native YUV (chroma not upsampled)')
parser.add_argument('-t', '--test', action='store_true', help='Test model and generate CSV submission file')
parser.add_argument('-v', '--validate', action='store_true', help='Validate CSV submission file')
parser.add_argument('-f', '--filename', default='eggs.csv', help='CSV file for submission')
parser.add_argument('-tf', '--test-files', nargs='*', help='List of test files')
parser.add_argument('-c', '--cpu', action='store_true', help='force CPU usage')
parser.add_argument('-i', '--idx', type=int, nargs='+', help='Indexes to use, e.g. -i 2 16')

args = parser.parse_args()

if args.cpu:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

if args.idx:
    IMGS_IDX = args.idx
else:
    IMGS_IDX = range(1,17)

IDXS     = len(IMGS_IDX)

SX   = 1920
SY   = 1280

TRAIN_FOLDER = 'train_hq'

ids = list(set([(x.split('/')[1]).split('_')[0] for x in glob.glob(join(TRAIN_FOLDER, '*_*.jpg'))]))

#ids_train = list(itertools.product(ids_train, IMGS_IDX))

load_mask  = lambda im, idx: imread(join('train_masks', '{}_{:02d}_mask.gif'.format(im, idx))).astype('uint8') # / 255. ).astype('float32') #/ 255.

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



