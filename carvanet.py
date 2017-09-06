import argparse
import glob
import numpy as np
import pandas as pd
import random
from scipy.misc import imread, imsave
from os.path import join
from sklearn.model_selection import train_test_split
from skimage.transform import rescale, downscale_local_mean
import skimage.exposure
import scipy.ndimage
from keras.optimizers import Adam, Adadelta, SGD
from keras.losses import binary_crossentropy
from keras import backend as K
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, LambdaCallback
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model, Model
from keras.layers import concatenate, Lambda
import itertools
import re
import os
import sys
import csv
from tqdm import tqdm
import jpeg4py as jpeg
from keras_contrib.layers import CRF

import densenet_fc as dc
import unet
from nadamaccum import * 

parser = argparse.ArgumentParser()
parser.add_argument('--max-epoch', type=int, default=200, help='Epoch to run')
parser.add_argument('-b', '--batch-size', type=int, default=1, help='Batch Size during training, e.g. -b 2')
parser.add_argument('-ba', '--batch-acc', type=int, default=1, help='Batch Size for training accumulation, e.g. -b 4')
parser.add_argument('-l', '--learning_rate', type=float, default=1e-3, help='Initial learning rate')
parser.add_argument('-m', '--model', help='load hdf5 model (and continue training)')
parser.add_argument('-s', '--scale', type=int, default=1, help='downscale e.g. -s 2')
parser.add_argument('-ar', '--rotation', type=float, default=0.5, help='Rotation angle for augmentation e.g. -ar 2')
parser.add_argument('-as', '--shift', action='store_true', help='Enable 1 pixel shifts')
parser.add_argument('-az', '--zoom', type=float, default=1., help='Zoom for augmentation e.g. -az 1.1')
parser.add_argument('-u', '--unet', action='store_true', help='use UNET')
parser.add_argument('-r', '--resnet', action='store_true', help='use residual dilated nets')
parser.add_argument('-yuv', '--yuv', action='store_true', help='Use native YUV (chroma not upsampled)')
parser.add_argument('-t', '--test', action='store_true', help='Test model and generate CSV submission file')
parser.add_argument('-tef', '--test-flip', action='store_true', help='Flip ensembling for testing')
parser.add_argument('-v', '--validate', action='store_true', help='Validate CSV submission file')
parser.add_argument('-f', '--filename', default='eggs.csv', help='CSV file for submission')
parser.add_argument('-tf', '--test-files', nargs='*', help='List of test files')
parser.add_argument('-tdr', '--test-dry-run', action='store_true', help='Dry run (only test first 100 cars)')
parser.add_argument('-tbe', '--test-bad-entries', action='store_true', help='Test bad RLE entries (debugging)')
parser.add_argument('-c', '--cpu', action='store_true', help='force CPU usage')
parser.add_argument('-i', '--idx', type=int, nargs='+', help='Indexes to use, e.g. -i 2 16')
parser.add_argument('-pt', '--half', action='store_true', help='Only do half image')
parser.add_argument('-ca', '--clahe', action='store_true', help='Contrast Limited Adaptive Histogram Equalization')

args = parser.parse_args()

if args.cpu:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

SX   = 1920
SY   = 1280
S    = args.scale

TRAIN_FOLDER = 'train_hq'
TEST_FOLDER  = 'test_hq'
MODEL_FOLDER = 'models'

if args.idx:
    IMGS_IDX = args.idx
else:
    IMGS_IDX = range(1,17)

IDXS     = len(IMGS_IDX)

IMGS_IDX_TO_FLIP = [] #range(10,17)

import scipy.stats as st
def gkern(kernlen=21, nsig=3):
    """Returns a 2D Gaussian kernel array."""

    interval = (2*nsig+1.)/(kernlen)
    x = np.linspace(-nsig-interval/2., nsig+interval/2., kernlen+1)
    kern1d = np.diff(st.norm.cdf(x))
    kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
    kernel = kernel_raw/kernel_raw.sum()
    return kernel

# From here: https://github.com/jocicmarko/ultrasound-nerve-segmentation/blob/master/train.py
def dice_coef(y_true, y_pred):
    smooth = 0.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def weighted_border_dice_coef(y_true, y_pred):
    k_width = 20
    w_max = 2.
    edge = K.abs(K.conv2d(y_true, np.float32([[[0,1,0],[1,-4,1],[0,1,0]]]).reshape((3,3,1,1)), padding='same', data_format='channels_last'))
    gk = w_max * np.ones((k_width,k_width, 1,1), dtype='float32') / 3.
    x_edge = K.clip(K.conv2d(edge, gk, padding='same', data_format='channels_last'), 0., w_max)
    w_f      = K.flatten(x_edge + 1.)
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(w_f * y_true_f * y_pred_f)
    return (2. * intersection ) / (K.sum(w_f * y_true_f) + K.sum(w_f * y_pred_f))

def weighted_dice_coef(y_true, y_pred):
    mean = 0.21649066
    w_1 = 1/mean**2
    w_0 = 1/(1-mean)**2
    y_true_f_1 = K.flatten(y_true)
    y_pred_f_1 = K.flatten(y_pred)
    y_true_f_0 = K.flatten(1-y_true)
    y_pred_f_0 = K.flatten(1-y_pred)

    intersection_0 = K.sum(y_true_f_0 * y_pred_f_0)
    intersection_1 = K.sum(y_true_f_1 * y_pred_f_1)

    return 2 * (w_0 * intersection_0 + w_1 * intersection_1) / ((w_0 * (K.sum(y_true_f_0) + K.sum(y_pred_f_0))) + (w_1 * (K.sum(y_true_f_1) + K.sum(y_pred_f_1))))

def dice_mask(y_true, y_pred):
    return 100 * dice_coef(y_true, K.round(y_pred))

def bce_dice_loss(y_true, y_pred):
    return 0.5 * binary_crossentropy(y_true, y_pred) + dice_coef_loss(y_true, y_pred)

def dice_coef_loss(y_true, y_pred):
    #return 1. - weighted_border_dice_coef(y_true, y_pred)
    return 100. * (1. - dice_coef(y_true, y_pred))
    #return weighted_bce_dice_loss(y_true, y_pred)

# from https://www.kaggle.com/lyakaap/weighing-boundary-pixels-loss-script-by-keras2
# weight: weighted tensor(same shape with mask image)
def weighted_bce_loss(y_true, y_pred, weight):
    # avoiding overflow
    epsilon = 1e-7
    y_pred = K.clip(y_pred, epsilon, 1. - epsilon)
    logit_y_pred = K.log(y_pred / (1. - y_pred))
    
    # https://www.tensorflow.org/api_docs/python/tf/nn/weighted_cross_entropy_with_logits
    loss = (1. - y_true) * logit_y_pred + (1. + (weight - 1.) * y_true) * \
    (K.log(1. + K.exp(-K.abs(logit_y_pred))) + K.maximum(-logit_y_pred, 0.))
    return K.sum(loss) / K.sum(weight)

def weighted_dice_loss(y_true, y_pred, weight):
    smooth = 1.
    w, m1, m2 = weight * weight, y_true, y_pred
    intersection = (m1 * m2)
    score = (2. * K.sum(w * intersection) + smooth) / (K.sum(w * m1) + K.sum(w * m2) + smooth)
    loss = 1. - K.sum(score)
    return loss

def weighted_bce_dice_loss(y_true, y_pred):
    y_true = K.cast(y_true, 'float32')
    y_pred = K.cast(y_pred, 'float32')
    # if we want to get same size of output, kernel size must be odd number
    averaged_mask = K.pool2d(
            y_true, pool_size=(11, 11), strides=(1, 1), padding='same', pool_mode='avg')
    border = K.cast(K.greater(averaged_mask, 0.005), 'float32') * K.cast(K.less(averaged_mask, 0.995), 'float32')
    weight = K.ones_like(averaged_mask)
    w0 = K.sum(weight)
    weight += border * 2
    w1 = K.sum(weight)
    weight *= (w0 / w1)
    loss = weighted_bce_loss(y_true, y_pred, weight) + \
    weighted_dice_loss(y_true, y_pred, weight)
    return loss

def gen(items, batch_size, training=True, inference=False, half=False, SX=SX, SY=SY):

    _SY = SY
    if half:
        SY = SY // 2

    X    = np.zeros((batch_size, SY//S, SX//S, 3), dtype=np.float32)

    if not inference:
        y    = np.zeros((batch_size, SY//S, SX//S, 1), dtype=np.float32)
    
    X_Y  = np.zeros((batch_size, SY//S,     SX//S,     1), dtype=np.float32)
    X_UV = np.zeros((batch_size, SY//(S*2), SX//(S*2), 2), dtype=np.float32)

    _img  = np.zeros((SY, SX, 3), dtype=np.float32)
    _mask = np.zeros((SY, SX, 1), dtype=np.float32)

    #data_gen_args = dict(
    #    rotation_range=args.rotation,
    #    zoom_range=0.)

    #image_datagen = ImageDataGenerator(**data_gen_args)
    
    assert batch_size <= 16
    input_folder = '.'

    load_img   = lambda im, idx: jpeg.JPEG(join(input_folder, TEST_FOLDER if inference else TRAIN_FOLDER, '{}_{:02d}.jpg'.format(im, idx))).decode()[:SY, ...].astype(np.float32) / 255.
    load_yuv   = lambda im, idx: jpeg.JPEG(join(input_folder, TEST_FOLDER if inference else TRAIN_FOLDER, '{}_{:02d}.jpg'.format(im, idx))).decodeYUV().astype(np.float32) / 255.
    load_mask  = lambda im, idx: imread(join(input_folder, 'train_masks', '{}_{:02d}_mask.gif'.format(im, idx)))[:SY, ...] / 255.
    mask_image = lambda im, mask: (im * np.expand_dims(mask, 2))

    batch_idx = 0
    seed_idx  = 0

    Xsum  = np.array([0.,0.,0.])
    Xmean = None
    Xvar  = np.array([0.,0.,0.])
    Xmoments_printed = False
    Ysum  = np.array([0.])
    Ymean = None

    while True:
        if training:
            random.shuffle(items)

        for _item in items:

            item, idx = _item
            can_rotate = False
            can_shift  = False
            can_flip   = False

            if not inference:
                mask = load_mask(item, idx)[...,0]

                mask_in_borders = np.sum(mask[:,0]) + np.sum(mask[:,-1]) + np.sum(mask[0,:]) + np.sum(mask[-1,:])
                can_rotate = (mask_in_borders == 0.) and (args.rotation != 0.) and training
                shift = np.random.randint(2, size=(2))
                can_shift  = (mask_in_borders == 0.) and training and (np.sum(shift) != 0.) and args.shift
                can_flip   = training # (idx in [1, 9]) and training
                flip = np.random.randint(2)
                rotate = (np.random.random() - 0.5) * args.rotation 

                if can_flip and flip:
                    mask = np.flip(mask, 1)

                if can_shift:
                    mask =  scipy.ndimage.interpolation.shift(mask, shift)
                mask = np.expand_dims(mask, axis=2)
                #scipy.ndimage.interpolation.rotate
                _mask[ :, :1918, :] = scipy.ndimage.interpolation.rotate(mask, rotate, reshape=False) if can_rotate else mask

                #_mask[ :, :1918, :] = image_datagen.random_transform(mask, seed=seed_idx) if can_rotate else mask
                _mask[ :, 1918:, :] = 0.

                if idx in IMGS_IDX_TO_FLIP:
                    _mask = np.flip(_mask, 1)

                y[batch_idx] = rescale(_mask, 1./S) if S != 1 else _mask

            if args.yuv:
                yuv  = load_yuv(item, idx)
                _y   = yuv[:SX*_SY].reshape((_SY,SX))[:SY,...]

                if args.clahe:
                    #print(np.amax(_y), np.amin(_y))
                    _y = skimage.exposure.equalize_adapthist(_y, kernel_size=64, clip_limit=0.02)
                    #print(np.amax(_y), np.amin(_y))

                if can_flip and flip:
                    _y = np.flip(_y, 1)
                if can_shift:
                    _y =  scipy.ndimage.interpolation.shift(_y, shift)
                _y = np.expand_dims(_y, axis=2)

                if yuv.shape[0] == SX*_SY + (2 * SX*_SY//4):
                    _u  = yuv[SX*_SY            : SX*_SY + SX*_SY//4].reshape((_SY//2, SX//2, 1))[:SY//2,...]
                    _v  = yuv[SX*_SY + SX*_SY//4 :                  ].reshape((_SY//2, SX//2, 1))[:SY//2,...]
                    _uv = np.concatenate((_u,_v), axis=2)
                    if can_flip and flip:
                        _uv = np.flip(_uv, 1)
                    if can_shift:
                        _uv = scipy.ndimage.interpolation.shift(_uv, np.hstack((shift/2.,[0])))

                elif yuv.shape[0] == SX*_SY*3:
                    _u  = yuv[SX*_SY:SX*_SY*2].reshape((_SY, SX, 1))[:SY,...]
                    _v  = yuv[SX*_SY*2:      ].reshape((_SY, SX, 1))[:SY,...]
                    _uv = np.concatenate((_u,_v), axis=2)
                    if can_flip and flip:
                        _uv = np.flip(_uv, 1)
                    if can_shift:
                        _uv = scipy.ndimage.interpolation.shift(_uv, np.hstack((shift,[0])))
                    _uv = downscale_local_mean(_uv, (2,2,1))                    

                else:
                    assert False

                #scipy.ndimage.interpolation.rotate(mask, rotate)
                _y   = scipy.ndimage.interpolation.rotate(_y,  rotate, reshape=False) if can_rotate else _y
                _uv  = scipy.ndimage.interpolation.rotate(_uv, rotate, reshape=False) if can_rotate else _uv
                #_y   = image_datagen.random_transform(_y,  seed=seed_idx) if can_rotate else _y
                #_uv  = image_datagen.random_transform(_uv, seed=seed_idx) if can_rotate else _uv

                _y [:,1918:,   :] = 0.                
                _uv[:,1918//2:,:] = 0.

                if idx in IMGS_IDX_TO_FLIP:
                    _y  = np.flip(_y,  1)
                    _uv = np.flip(_uv, 1)

                X_Y[batch_idx]  = rescale(_y,  1./S) if S != 1 else _y
                X_UV[batch_idx] = rescale(_uv, 1./S) if S != 1 else _uv

                del yuv

            else:
                img = load_img(item, idx)

                _img[:, :1918, :] = image_datagen.random_transform(img, seed=seed_idx) if training and (mask_in_borders == 0.) else img
                _img[:, 1918:, :] = 0.

                if idx in IMGS_IDX_TO_FLIP:
                    _img = np.flip(_img, 1)

                X[batch_idx] = rescale(_img, 1./S) if S != 1 else _img

            seed_idx  += 1
            batch_idx += 1

            if batch_idx == batch_size:

                if args.yuv:
                    #pass
                    #X = X / np.array([ 0.2466268 ,  0.02347598,  0.02998368]) - np.array([  2.8039049 ,  21.16614256,  16.76252866])
                    X_Y  /= 0.2466268
                    X_Y  -= 2.8039049

                    X_UV /= np.float32([  0.02347598,  0.02998368 ]) 
                    X_UV -= np.float32([ 21.16614256, 16.76252866 ])

                else:
                    X = X / np.float32([ 0.23165472,  0.23369996,  0.23183985]) - np.float32([ 3.13899873,  3.12144822,  3.11967396])

                if not inference:
                    y[ y >= 0.5] = 1.
                    y[ y <  0.5] = 0.

                __X = [X_Y,X_UV] if args.yuv else X

                if not inference:
                    yield(__X, y)
                else:   
                    yield(__X)

                batch_idx = 0
                if (seed_idx % 10) == 0 :
                    for b in range(batch_size):
                        if args.yuv:
                            imsave("_i_Y"+str(b)+".jpg",  X_Y[b,...,0])
                            imsave("_i_U"+str(b)+".jpg", X_UV[b,...,0])
                            imsave("_i_V"+str(b)+".jpg", X_UV[b,...,1])
                        else:
                            imsave("_i"+str(b)+".jpg", X[b])
                        if not inference:
                            imsave("_m"+str(b)+".jpg", y[b,...,0])

                if not args.yuv:
                    Xsum += X.mean(axis=(0,1,2))
                    if Xmean is not None:
                        Xvar += ((X - Xmean) ** 2).mean(axis=(0,1,2))

                if not inference:
                    Ysum += y.mean(axis=(0,1,2))

        if Xmoments_printed is False and not args.yuv:
            if Xmean is None:
                Xmean = Xsum / (len(items) / batch_size)
                print("Xmean: ", Xmean)
                if not inference:
                    Ymean = Ysum / (len(items) / batch_size)
                    print("Ymean: ", Ymean)
            else:
                Xvar /= (len(items) / batch_size)
                print("Xvar: ", Xvar)
                print("Scale: 1./", np.sqrt(Xvar), "Offset: -", Xmean / np.sqrt(Xvar))
                Xmoments_printed = True

        Xsum = 0.

if args.model:
    print("Loading model " + args.model)

    # monkey-patch loss so model loads ok
    # https://github.com/fchollet/keras/issues/5916#issuecomment-290344248
    import keras.losses
    import keras.metrics
    keras.losses.bce_dice_loss = bce_dice_loss
    keras.losses.dice_coef_loss = dice_coef_loss
    keras.metrics.dice_coef = dice_coef
    keras.metrics.dice_mask = dice_mask
    keras.optimizers.AdamAccum = AdamAccum

    model = load_model(args.model, compile=False)
    match = re.search(r'([a-z]+)(_([0-9]+)_([0-9]+))?-s\d-epoch(\d+)-.*\.hdf5', args.model)
    model_name = match.group(1)
    last_epoch = int(match.group(5)) + 1
    if match.group(2):
        IMGS_IDX = [int(match.group(3)), int(match.group(4))]
        IDXS     = len(IMGS_IDX)
        assert args.idx == None

else:
    last_epoch = 0

    if args.unet:
        if args.yuv:
            model = unet.U3Net((SY//S, SX//S), activation='sigmoid', int_activation='relu', yuv=True, half=args.half)
        else:           
            model = unet.U3Net((SY//S, SX//S, 3), activation='sigmoid', int_activation='relu', yuv=False)
        model_name = 'unet'
    elif args.resnet:
        model = unet.RNet((SY//S, SX//S), activation='sigmoid', int_activation='relu', yuv=True, dropout_rate=0.05)
        model_name = 'rnet'
    else:
        model = dc.DenseNetFCN((SY//S, SX//S, 3), yuv=args.yuv, nb_dense_block=4, growth_rate=8, dropout_rate = 0.2,
        	nb_layers_per_block=3, upsampling_type='deconv', classes=1, activation='sigmoid', init_conv_filters=32,
        	batch_norm=False, int_activation='relu', batchsize=args.batch_size)
        model_name = 'fc-densenet'

ids = list(set([(x.split('/')[1]).split('_')[0] for x in glob.glob(join(TRAIN_FOLDER,'*_*.jpg'))]))
ids.sort()

ids_train, ids_val = train_test_split(ids, test_size=0.1, random_state=42)

ids_train = list(itertools.product(ids_train, IMGS_IDX))
ids_val   = list(itertools.product(ids_val,   IMGS_IDX))

model.summary()
#opt = NadamAccum(lr=args.learning_rate, accum_iters=32)
#opt = Adam(lr=args.learning_rate)
opt = AdamAccum(lr=args.learning_rate, accumulator=args.batch_acc)
#opt = SGD(lr=args.learning_rate, decay=1e-6, momentum=0.9, nesterov=True)

model.compile(optimizer=opt, loss=dice_coef_loss, metrics=['accuracy', dice_coef, dice_mask])

def rle_encode(pixels):
    pixels = pixels[:, :1918,:]
    pixels = pixels.ravel()
    np.rint(pixels, out=pixels)
    
    # We avoid issues with '1' at the start or end (at the corners of 
    # the original image) by setting those pixels to '0' explicitly.
    # We do not expect these to be non-zero for an accurate mask, 
    # so this should not harm the score.
    pixels[0]  = 0
    pixels[-1] = 0
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 2
    runs[1::2] = runs[1::2] - runs[:-1:2]
    return runs

def return_bad_rle_entries(mask):
    rr = 0
    for m in itertools.izip(*[itertools.islice(mask, j, None, 2) for j in range(2)]):
        y = m[0] // 1918
        x = m[0]  % 1918
        r = m[1]
        rr += r
        assert y < SY
        if r > SX*SY//2:
            return (m, x, y, r)
    if rr > 3*SX*SY//4:
        return(rr,)
    return []

def rle_to_string(runs):
    return ' '.join(str(x) for x in runs)

if args.test:
    if args.test_files:
        ids_test = [[x.split('_')[0], int(x.split('_')[1])] for x in args.test_files]
    else:
        _ids_test = list(set([(x.split('/')[1]).split('_')[0] for x in glob.glob(join(TEST_FOLDER, '*_*.jpg'))]))
        if args.test_dry_run:
            _ids_test = _ids_test[:100]
        ids_test = list(itertools.product(_ids_test, IMGS_IDX))

    generator = gen(ids_test, args.batch_size, training = False, inference = True)

    with open(args.filename, 'wb') as csvfile, open("flip-" + args.filename, 'wb') as csvfile_flip:
        writer = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(['img', 'rle_mask'])
        if args.test_flip:
            writer_flip = csv.writer(csvfile_flip, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            writer_flip.writerow(['img', 'rle_mask'])
        assert len(ids_test) % args.batch_size == 0
        id_generator = itertools.izip(*[itertools.islice(ids_test, j, None, args.batch_size) for j in range(args.batch_size)])
        for i in tqdm(range(len(ids_test) // args.batch_size)):
            ids = next(id_generator)
            mini_batch = next(generator)
            prediction_masks = model.predict_on_batch(mini_batch)
            if args.test_flip:
                #print(len(mini_batch))
                #print(mini_batch[1].shape)
                _prediction_masks = model.predict_on_batch([np.flip(mini_batch[0], 1), np.flip(mini_batch[1], 1)])
                _prediction_masks = np.flip(_prediction_masks, 1)
                prediction_masks_flipped_ensemble = (prediction_masks + _prediction_masks) / 2.
            else:
                prediction_masks_flipped_ensemble = prediction_masks
            for idx,prediction_mask, prediction_mask_flipped_ensemble in zip(ids, prediction_masks, prediction_masks_flipped_ensemble):
                fname = '{}_{:02d}.jpg'.format(idx[0], idx[1])

                if idx[1] in IMGS_IDX_TO_FLIP:
                    prediction_mask = np.flip(prediction_mask, 1)

                if args.test_files:
                    imsave('pred_'+fname, prediction_mask[...,0])

                rle = rle_encode(prediction_mask)
                writer.writerow([fname, rle_to_string(rle)])

                if args.test_flip:
                    rle_flipped_ensemble = rle_encode(prediction_mask_flipped_ensemble)
                    writer_flip.writerow([fname, rle_to_string(rle_flipped_ensemble)])

                if args.test_bad_entries:
                    bad_entries = return_bad_rle_entries(rle)
                    if bad_entries:
                        print(fname)
                        print(bad_entries)
                        imsave('bad_'+fname, prediction_mask[...,0])


elif args.validate:
    csv.field_size_limit(sys.maxsize)
    with open(args.filename, 'rb') as csvfile:

        reader = csv.reader(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        assert next(reader) == ['img', 'rle_mask']
        for f in reader:
            assert len(f) == 2
            file = f[0]
            mask = map(int, f[1].split(' '))
            bad_file = False
            bad_entries = return_bad_rle_entries(mask)
            if bad_entries:
                print(file)
                print(bad_entries)

else:
    metric  = "-val_dice_coef{val_dice_coef:.6f}"
    monitor = 'val_dice_coef'
    idx = "" if IMGS_IDX == range(1,17) else "_" + "_".join(map(str,IMGS_IDX))

    save_checkpoint = ModelCheckpoint(
            join(MODEL_FOLDER, model_name+idx+"-s"+str(S)+"-epoch{epoch:02d}"+metric+".hdf5"),
            monitor=monitor,
            verbose=0,  save_best_only=True, save_weights_only=False, mode='max', period=1)

    reduce_lr = ReduceLROnPlateau(monitor=monitor, factor=0.5, patience=4, min_lr=1e-7, epsilon = 0.00001, verbose=1, mode='max')

    # Function to display the target and prediciton
    def testmodel(epoch, logs):
        predx, predy = next(gen(ids_val, args.batch_size, training = False, half=args.half))
        
        predout = model.predict(
            predx,
            batch_size=args.batch_size
        )
        for b in range(args.batch_size):
            imsave("GT_e"+str(epoch)+"_"+str(b)+".jpg",   predy[b,...,0])
            imsave("PR_e"+str(epoch)+"_"+str(b)+".jpg", predout[b,...,0])

    # Callback to display the target and prediciton
    testmodelcb = LambdaCallback(on_epoch_end=testmodel)

    model.fit_generator(
            generator        = gen(ids_train, args.batch_size, half=args.half),
            steps_per_epoch  = len(ids_train)  // args.batch_size,
            validation_data  = gen(ids_val, args.batch_size, training = False, half=args.half),
            validation_steps = len(ids_val) // args.batch_size,
            epochs = args.max_epoch,
            callbacks = [save_checkpoint, reduce_lr, testmodelcb],
            initial_epoch = last_epoch, max_queue_size=1, workers=1)


