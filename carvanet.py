import argparse
import glob
import numpy as np
import pandas as pd
import random
from scipy.misc import imread, imsave
from os.path import join
from sklearn.model_selection import train_test_split
from skimage.transform import rescale, downscale_local_mean
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
#from crfrnn_layer import CrfRnnLayer as CRF
from keras_contrib.layers import CRF

import densenet_fc as dc
import unet
from nadamaccum import * 

parser = argparse.ArgumentParser()
parser.add_argument('--max-epoch', type=int, default=200, help='Epoch to run')
parser.add_argument('-b', '--batch-size', type=int, default=1, help='Batch Size during training, e.g. -b 2')
parser.add_argument('-l', '--learning_rate', type=float, default=1e-3, help='Initial learning rate')
parser.add_argument('-m', '--model', help='load hdf5 model (and continue training)')
parser.add_argument('-s', '--scale', type=int, default=1, help='downscale e.g. -s 2')
parser.add_argument('-u', '--unet', action='store_true', help='use UNET')
parser.add_argument('-r', '--resnet', action='store_true', help='use residual dilated nets')
parser.add_argument('-yuv', '--yuv', action='store_true', help='Use native YUV (chroma not upsampled)')
parser.add_argument('-t', '--test', action='store_true', help='Test model and generate CSV submission file')
parser.add_argument('-v', '--validate', action='store_true', help='Validate CSV submission file')
parser.add_argument('-f', '--filename', default='eggs.csv', help='CSV file for submission')
parser.add_argument('-tf', '--test-files', nargs='*', help='List of test files')
parser.add_argument('-crf', '--crf', action='store_true', help='Add CRF layer')
parser.add_argument('-c', '--cpu', action='store_true', help='force CPU usage')
parser.add_argument('-i', '--idx', type=int, nargs='+', help='Indexes to use, e.g. -i 2 16')

args = parser.parse_args()

if args.cpu:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

SX   = 1920
SY   = 1280
S    = args.scale

if args.idx:
    IMGS_IDX = args.idx
else:
    IMGS_IDX = range(1,17)

IDXS     = len(IMGS_IDX)

IMGS_IDX_TO_FLIP = range(10,17)

# From here: https://github.com/jocicmarko/ultrasound-nerve-segmentation/blob/master/train.py
def dice_coef(y_true, y_pred):
    smooth = 0.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

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
    return dice_coef(y_true, K.round(y_pred))

def bce_dice_loss(y_true, y_pred):
    return 0.5 * binary_crossentropy(y_true, y_pred) + dice_coef_loss(y_true, y_pred)

def dice_coef_loss(y_true, y_pred):
    return 1. - weighted_dice_coef(y_true, y_pred)

def gen(items, batch_size, training=True, inference=False):
    X    = np.zeros((batch_size, SY//S, SX//S, 3), dtype=np.float32)

    if not inference:
        y    = np.zeros((batch_size, SY//S, SX//S, 1), dtype=np.float32)
    
    X_Y  = np.zeros((batch_size, SY//S,     SX//S,     1), dtype=np.float32)
    X_UV = np.zeros((batch_size, SY//(S*2), SX//(S*2), 2), dtype=np.float32)

    _img  = np.zeros((SY, SX, 3), dtype=np.float32)
    _mask = np.zeros((SY, SX, 1), dtype=np.float32)

    data_gen_args = dict(
        rotation_range=1.,
        zoom_range=0.)

    image_datagen = ImageDataGenerator(**data_gen_args)
    
    assert batch_size <= 16
    input_folder = '.'

    #load_img   = lambda im, idx: imread(join(input_folder, 'test' if inference else 'train', '{}_{:02d}.jpg'.format(im, idx)), mode='YCbCr') / 255.
    load_img   = lambda im, idx: jpeg.JPEG(join(input_folder, 'test' if inference else 'train', '{}_{:02d}.jpg'.format(im, idx))).decode().astype(np.float32) / 255.
    load_yuv   = lambda im, idx: jpeg.JPEG(join(input_folder, 'test' if inference else 'train', '{}_{:02d}.jpg'.format(im, idx))).decodeYUV().astype(np.float32) / 255.
    load_mask  = lambda im, idx: imread(join(input_folder, 'train_masks', '{}_{:02d}_mask.gif'.format(im, idx))) / 255.
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

            if not inference:
                mask = load_mask(item, idx)[...,:1]

                mask_in_borders = np.sum(mask[:,0]) + np.sum(mask[:,-1]) + np.sum(mask[0,:]) + np.sum(mask[-1,:])

                _mask[ :, :1918, :] = image_datagen.random_transform(mask, seed=seed_idx) if training and (mask_in_borders == 0.) else mask
                _mask[ :, 1918:, :] = 0.

                if idx in IMGS_IDX_TO_FLIP:
                    _mask = np.flip(_mask, 1)

                y[batch_idx] = rescale(_mask, 1./S) if S != 1 else _mask

            if args.yuv:
                yuv  = load_yuv(item, idx)
                _y   = yuv[:SX*SY].reshape((SY,SX,1))
                if yuv.shape[0] == SX*SY + (2 * SX*SY//4):
                    _u  = yuv[SX*SY            : SX*SY + SX*SY//4].reshape((SY//2, SX//2, 1))
                    _v  = yuv[SX*SY + SX*SY//4 :                 ].reshape((SY//2, SX//2, 1))
                    _uv = np.concatenate((_u,_v), axis=2)
                elif yuv.shape[0] == SX*SY*3:
                    _u  = yuv[SX*SY:SX*SY*2].reshape((SY, SX, 1))
                    _v  = yuv[SX*SY*2:     ].reshape((SY, SX, 1))
                    _u  = downscale_local_mean(_u, (2,2,1))
                    _v  = downscale_local_mean(_v, (2,2,1))
                    _uv = np.concatenate((_u,_v), axis=2)
                else:
                    assert False

                _y   = image_datagen.random_transform(_y,  seed=seed_idx) if training and (mask_in_borders == 0.) else _y
                _uv  = image_datagen.random_transform(_uv, seed=seed_idx) if training and (mask_in_borders == 0.) else _uv

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
                # X = X / np.array([ 0.2466268 ,  0.02347598,  0.02998368]) - np.array([  2.8039049 ,  21.16614256,  16.76252866])
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

ids = list(set([(x.split('_')[0]).split('/')[1] for x in glob.glob('train/*_*.jpg')]))

ids_train, ids_val = train_test_split(ids, test_size=0.1, random_state=42)

ids_train = list(itertools.product(ids_train, IMGS_IDX))
ids_val   = list(itertools.product(ids_val,   IMGS_IDX))

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

    model = load_model(args.model)
    match = re.search(r'([a-z]+)-s\d-epoch(\d+)-.*\.hdf5', args.model)
    model_name = match.group(1)
    last_epoch = int(match.group(2)) + 1

else:
    last_epoch = 0

    if args.unet:
        if args.yuv:
            model = unet.U3Net((SY//S, SX//S), activation='sigmoid', int_activation='relu', yuv=True, dropout_rate=0.05)
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

if args.crf:
    unary = model.layers[-1].output
    print(unary)
    unary = concatenate([Lambda(lambda x: 1.-x)(unary), unary], axis=3, name='unaries')
    print(unary)
    print(model.inputs[0])

    crf   = CRF(image_dims=(SY//S, SX//S),
                         num_classes=2,
                         theta_alpha=160.,
                         theta_beta=3.,
                         theta_gamma=3.,
                         num_iterations=10,
                         batch_size = args.batch_size,
                         name='crfrnn')([unary, model.layers[0].output])
    print(crf)
    output = crf
    output = Lambda(lambda x: x[...,1:2])(crf)
    print(output)
    model  = Model(inputs=model.inputs, outputs=[output])

model.summary()
#opt = NadamAccum(lr=args.learning_rate, accum_iters=32)
opt = Adam(lr=args.learning_rate)
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
        _ids_test = list(set([(x.split('_')[0]).split('/')[1] for x in glob.glob('test/*_*.jpg')]))
        ids_test = list(itertools.product(_ids_test, IMGS_IDX))

    generator = gen(ids_test, args.batch_size, training = False, inference = True)

    with open(args.filename, 'wb') as csvfile:
        writer = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(['img', 'rle_mask'])
        assert len(ids_test) % args.batch_size == 0
        id_generator = itertools.izip(*[itertools.islice(ids_test, j, None, args.batch_size) for j in range(args.batch_size)])
        for i in tqdm(range(len(ids_test) // args.batch_size)):
            ids = next(id_generator)
            prediction_masks = model.predict_on_batch(next(generator))
            for idx,prediction_mask  in zip(ids, prediction_masks):
                fname = '{}_{:02d}.jpg'.format(idx[0], idx[1])

                if idx[1] in IMGS_IDX_TO_FLIP:
                    prediction_mask = np.flip(prediction_mask, 1)

                if args.test_files:
                    imsave('pred_'+fname, prediction_mask[...,0])

                rle = rle_encode(prediction_mask)
                bad_entries = return_bad_rle_entries(rle)
                if bad_entries:
                    print(fname)
                    print(bad_entries)
                    imsave('bad_'+fname, prediction_mask[...,0])

                writer.writerow([fname, rle_to_string(rle)])

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
    metric  = "-val_dice_coef{val_dice_coef:.4f}"
    monitor = 'val_dice_coef'
    idx = "" if IMGS_IDX == range(1,17) else "_" + "_".join(map(str,IMGS_IDX))

    save_checkpoint = ModelCheckpoint(
            model_name+idx+"-s"+str(S)+"-epoch{epoch:02d}"+metric+".hdf5",
            monitor=monitor,
            verbose=0,  save_best_only=True, save_weights_only=False, mode='max', period=1)

    reduce_lr = ReduceLROnPlateau(monitor=monitor, factor=0.5, patience=4, min_lr=1e-7, epsilon = 0.00001, verbose=1, mode='max')

    # Function to display the target and prediciton
    def testmodel(epoch, logs):
        predx, predy = next(gen(ids_val, args.batch_size, training = False))
        
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
            generator        = gen(ids_train, args.batch_size),
            steps_per_epoch  = len(ids_train)  // args.batch_size,
            validation_data  = gen(ids_val, args.batch_size, training = False),
            validation_steps = len(ids_val) // args.batch_size,
            epochs = args.max_epoch,
            callbacks = [save_checkpoint, reduce_lr, testmodelcb],
            initial_epoch = last_epoch, max_queue_size=1, workers=1)


