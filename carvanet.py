import argparse
import glob
import numpy as np
import pandas as pd
import random
from scipy.misc import imread, imsave
from os.path import join
from sklearn.model_selection import train_test_split
from skimage.transform import rescale, downscale_local_mean
from keras.optimizers import Adam, Adadelta
from keras.losses import binary_crossentropy
from keras import backend as K
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
import itertools
import re
import csv


import densenet_fc as dc
import unet

parser = argparse.ArgumentParser()
parser.add_argument('--max-epoch', type=int, default=200, help='Epoch to run')
parser.add_argument('-b', '--batch-size', type=int, default=1, help='Batch Size during training, e.g. -b 2')
parser.add_argument('-l', '--learning_rate', type=float, default=1e-3, help='Initial learning rate')
parser.add_argument('-m', '--model', help='load hdf5 model (and continue training)')
parser.add_argument('-s', '--scale', type=int, default=1, help='downscale e.g. -s 2')
parser.add_argument('-u', '--unet', action='store_true', help='use UNET')
parser.add_argument('-yuv', '--yuv', action='store_true', help='Use native YUV with 4:2:2 downsampling')
parser.add_argument('-t', '--test', action='store_true', help='Test model')

args = parser.parse_args()


SX   = 1920
SY   = 1280
S    = args.scale

IMGS_IDX = range(1,17)
IDXS     = len(IMGS_IDX)

# From here: https://github.com/jocicmarko/ultrasound-nerve-segmentation/blob/master/train.py
def dice_coef(y_true, y_pred):
    smooth = 0.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def bce_dice_loss(y_true, y_pred):
    return 0.5 * binary_crossentropy(y_true, y_pred) - dice_coef(y_true, y_pred)

def dice_coef_loss(y_true, y_pred):
    return 1. - dice_coef(y_true, y_pred)

def gen(items, batch_size, training=True, inference=False):
    X    = np.zeros((batch_size, SY//S, SX//S, 3), dtype=np.float32)

    if not inference:
        y    = np.zeros((batch_size, SY//S, SX//S, 1), dtype=np.float32)
    
    X_Y  = np.zeros((batch_size, SY//S,     SX//S,     1), dtype=np.float32)
    X_UV = np.zeros((batch_size, SY//(S*2), SX//(S*2), 2), dtype=np.float32)

    _img  = np.zeros((SY, SX, 3), dtype=np.float32)
    _mask = np.zeros((SY, SX, 1), dtype=np.float32)

    data_gen_args = dict(
        rotation_range=0.5,
        width_shift_range=0.05,
        height_shift_range=0.05,
        horizontal_flip=False,
        zoom_range=0.)

    image_datagen = ImageDataGenerator(**data_gen_args)
    
    assert batch_size <= 16
    input_folder = '.'

    load_img   = lambda im, idx: imread(join(input_folder, 'test' if inference else 'train', '{}_{:02d}.jpg'.format(im, idx)), mode='YCbCr')
    load_mask  = lambda im, idx: imread(join(input_folder, 'train_masks', '{}_{:02d}_mask.gif'.format(im, idx)))
    mask_image = lambda im, mask: (im * np.expand_dims(mask, 2))

    batch_idx = 0
    seed_idx  = 0

    Xsum  = np.array([0.,0.,0.])
    Xmean = None
    Xvar  = np.array([0.,0.,0.])
    Xmoments_printed = False

    imgs_idx_to_flip = range(10,17)

    while True:
        if training:
            random.shuffle(items)

        for _item in items:

            item, idx = _item

            if not inference:
                mask = load_mask(item, idx)[...,:1] / 255.
                if idx in imgs_idx_to_flip:
                    mask = np.flip(mask, 1)

                mask_in_borders = np.sum(mask[:,0]) + np.sum(mask[:,-1]) + np.sum(mask[0,:]) + np.sum(mask[-1,:])
                if mask_in_borders:
                    print(item)

                _mask[ :, :1918, :] = image_datagen.random_transform(mask, seed=seed_idx) if training else mask
                _mask[ :, 1918:, :] = 0.
                y[batch_idx] = rescale(_mask, 1./S) if S != 1 else _mask

            img = load_img(item, idx) / 255.
            if idx in imgs_idx_to_flip:
                img = np.flip(img, 1)

            _img[:, :1918, :] = image_datagen.random_transform(img, seed=seed_idx) if training else img
            _img[:, 1918:, :] = 0.
            X[batch_idx] = rescale(_img, 1./S) if S != 1 else _img

            seed_idx  += 1
            batch_idx += 1

            if batch_idx == batch_size:

                _X = X / np.array([ 0.2466268 ,  0.02347598,  0.02998368]) - np.array([  2.8039049 ,  21.16614256,  16.76252866])

                #X = X / np.array([ 0.23165472,  0.23369996,  0.23183985]) - np.array([ 3.13899873,  3.12144822,  3.11967396])

                if not inference:
                    y[ y >= 0.5] = 1.
                    y[ y <  0.5] = 0.

                if args.yuv:
                    X_Y  = _X[...,:1]
                    X_UV = downscale_local_mean(_X[...,1:3], (1,2,2,1))

                __X = [X_Y,X_UV] if args.yuv else _X

                if not inference:
                    yield(__X, y)
                else:   
                    yield(__X)

                batch_idx = 0
                if (seed_idx % 10) == 0 and not inference:
                    for b in range(batch_size):
                        imsave("_i"+str(b)+".jpg", X[b])
                        imsave("_m"+str(b)+".jpg", y[b,...,0])

                Xsum += X.mean(axis=(0,1,2))
                if Xmean is not None:
                    Xvar += ((X - Xmean) ** 2).mean(axis=(0,1,2))

        if Xmoments_printed is False:
            if Xmean is None:
                Xmean = Xsum / len(items)
                print("Xmean: ", Xmean)
            else:
                Xvar /= len(items) 
                print("Xvar: ", Xvar)
                print("Scale: 1./", np.sqrt(Xvar), "Offset: -", Xmean / np.sqrt(Xvar))
                Xmoments_printed = True

        Xsum = 0.

ids = list(set([(x.split('_')[0]).split('/')[1] for x in glob.glob('train/*_*.jpg')]))

ids_train, ids_val = train_test_split(ids, test_size=0.1)

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

    model = load_model(args.model)
    match = re.search(r'([a-z\-]+)-s\d.*\.hdf5', args.model)
    model_name = match.group(1)

else:

    if args.unet:
        if args.yuv:
            model = unet.U2Net((SY//S, SX//S, 3), activation='sigmoid', int_activation='relu')
        else:           
            model = unet.UNet((SY//S, SX//S, 3), activation='sigmoid', int_activation='relu')
        model_name = 'unet'
    else:
        model = dc.DenseNetFCN((SY//S, SX//S, 3), nb_dense_block=4, growth_rate=16, dropout_rate = 0.2,
        	nb_layers_per_block=4, upsampling_type='deconv', classes=1, activation='sigmoid', 
        	batch_norm=False, int_activation='relu', batchsize=args.batch_size)
        model_name = 'fc-densenet'


model.summary()

model.compile(Adam(lr=args.learning_rate), loss='binary_crossentropy', metrics=['accuracy', dice_coef])

def rle_encode(pixels):
    pixels = pixels.ravel()
    np.rint(pixels, out=pixels)
    
    # We avoid issues with '1' at the start or end (at the corners of 
    # the original image) by setting those pixels to '0' explicitly.
    # We do not expect these to be non-zero for an accurate mask, 
    # so this should not harm the score.
    pixels[0] = 0
    pixels[-1] = 0
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 2
    runs[1::2] = runs[1::2] - runs[:-1:2]
    runs[0::2] -= (runs[0::2]//1920) * 2
    return runs

def rle_to_string(runs):
    return ' '.join(str(x) for x in runs)

if args.test:
    _ids_test = list(set([(x.split('_')[0]).split('/')[1] for x in glob.glob('test/*_*.jpg')]))
    ids_test = list(itertools.product(_ids_test, IMGS_IDX))

    generator = gen(ids_test, args.batch_size, training = False, inference = True)

    with open('eggs.csv', 'wb') as csvfile:
        writer = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(['img', 'rle_mask'])
        i = 0
        for ids in itertools.izip(*[itertools.islice(ids_test, j, None, args.batch_size) for j in range(args.batch_size)]):
            prediction_masks = model.predict_generator(generator, steps=1, verbose=0 )
            print(ids, len(prediction_masks))
            for idx,prediction_mask  in zip(ids, prediction_masks):
                rle = rle_to_string(rle_encode(prediction_mask))
                fname = '{}_{:02d}.jpg'.format(idx[0], idx[1])
                writer.writerow([fname, rle])
                print(str(i) + "/" + str(len(ids_test)))
                i += 1


else:
    metric  = "-val_dice_coef{val_dice_coef:.4f}"
    monitor = 'val_dice_coef'

    save_checkpoint = ModelCheckpoint(
            model_name+"-s"+str(S)+"-epoch{epoch:02d}"+metric+".hdf5",
            monitor=monitor,
            verbose=0,  save_best_only=True, save_weights_only=False, mode='max', period=1)

    reduce_lr = ReduceLROnPlateau(monitor=monitor, factor=0.5, patience=20, min_lr=1e-7, epsilon = 0.0001, verbose=1)

    # Function to display the target and prediciton
    def testmodel(epoch, logs):
        predx, predy = next(gen(ids_val, args.batch_size, training = False))
        
        predout = model.predict(
            predx,
            batch_size=args.batch_size
        )
        for b in args.batch_size:
            imsave("GT_"+str(b)+".jpg", predy[b,...,0])
            imsave("PR_"+str(b)+".jpg", predout[b,...,0])

    # Callback to display the target and prediciton
    testmodelcb = keras.callbacks.LambdaCallback(on_epoch_end=testmodel)

    model.fit_generator(
            generator        = gen(ids_train, args.batch_size),
            steps_per_epoch  = len(ids_train)  // args.batch_size,
            validation_data  = gen(ids_val, args.batch_size, training = False),
            validation_steps = len(ids_val) // args.batch_size,
            epochs = args.max_epoch,
            callbacks = [save_checkpoint, reduce_lr, visualize_predictions, testmodelcb])


