import argparse
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from scipy.misc import imread, imsave
from os.path import join
from sklearn.model_selection import train_test_split
from keras.optimizers import Adam
from keras.losses import binary_crossentropy
from keras import backend as K
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model


import densenet_fc as dc

parser = argparse.ArgumentParser()
parser.add_argument('--max-epoch', type=int, default=200, help='Epoch to run')
parser.add_argument('-b', '--batch-size', type=int, default=1, help='Batch Size during training, e.g. -b 2')
parser.add_argument('-l', '--learning_rate', type=float, default=1e-3, help='Initial learning rate')
parser.add_argument('-m', '--model', help='load hdf5 model (and continue training)')
args = parser.parse_args()

SX = 1920
SY = 1280

# From here: https://github.com/jocicmarko/ultrasound-nerve-segmentation/blob/master/train.py
def dice_coef(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def bce_dice_loss(y_true, y_pred):
    return 0.5 * binary_crossentropy(y_true, y_pred) - dice_coef(y_true, y_pred)

def gen(items, batch_size, training=True):
    X = np.zeros((batch_size, SY, SX, 3), dtype=np.float32)
    y = np.zeros((batch_size, SY, SX, 1), dtype=np.float32)

    data_gen_args = dict(rotation_range=10.,
                     width_shift_range=0.1,
                     height_shift_range=0.1,
                     zoom_range=0.2)

    image_datagen = ImageDataGenerator(**data_gen_args)
    
    assert batch_size <= 16
    input_folder = '.'

    load_img   = lambda im, idx: imread(join(input_folder, 'train', '{}_{:02d}.jpg'.format(im, idx)))
    load_mask  = lambda im, idx: imread(join(input_folder, 'train_masks', '{}_{:02d}_mask.gif'.format(im, idx)))
    mask_image = lambda im, mask: (im * np.expand_dims(mask, 2))

    batch_idx = 0
    seed_idx  = 0

    while True:
        if training:
            random.shuffle(items)

        for item in items:

            imgs_idx   = list(range(1, 17))
            if training:
                random.shuffle(imgs_idx)

            for idx in imgs_idx:

                img = load_img(item, idx) / 255.
                X[batch_idx, :, :1918, :] = image_datagen.random_transform(img, seed=seed_idx) if training else img
                X[batch_idx, :, 1918:, :] = 0.

                img = load_mask(item, idx)[...,:1] / 255.
                y[batch_idx, :, :1918, :] = image_datagen.random_transform(img, seed=seed_idx) if training else img
                y[batch_idx, :, 1918:, :] = 0.

                seed_idx += 1
                batch_idx += 1

                if batch_idx == batch_size:
                    X -= 0.5
                    yield(X, y)
                    batch_idx = 0

ids = list(set([(x.split('_')[0]).split('/')[1] for x in glob.glob('train/*_*.jpg')]))

ids_train, ids_val = train_test_split(ids, test_size=0.2, random_state=42)

if args.model:
    print("Loading model " + args.model)

    # monkey-patch loss so model loads ok
    # https://github.com/fchollet/keras/issues/5916#issuecomment-290344248
    import keras.losses
    import keras.metrics
    keras.losses.bce_dice_loss = bce_dice_loss
    keras.metrics.dice_coef = dice_coef

    model = load_model(args.model)

else:

    model = dc.DenseNetFCN((SY, SX, 3), nb_dense_block=5, growth_rate=6, 
    	nb_layers_per_block=4, upsampling_type='upsampling', classes=1, activation='sigmoid', 
    	batch_norm=False, int_activation='relu')

model.summary()

model.compile(Adam(lr=args.learning_rate), bce_dice_loss, metrics=['accuracy', dice_coef])

metric  = "-val_dice_coef{val_dice_coef:.4f}"
monitor = 'val_dice_coef'

save_checkpoint = ModelCheckpoint(
        "fc-densenet-epoch{epoch:02d}"+metric+".hdf5",
        monitor=monitor,
        verbose=0,  save_best_only=False, save_weights_only=False, mode='auto', period=1)

reduce_lr = ReduceLROnPlateau(monitor=monitor, factor=0.5, patience=10, min_lr=1e-7, epsilon = 0.0001, verbose=1)





model.fit_generator(
        generator        = gen(ids_train, args.batch_size),
        steps_per_epoch  = len(ids_train) * 16 // args.batch_size,
        validation_data  = gen(ids_val, args.batch_size, training = False),
        validation_steps = len(ids_val) * 16 // args.batch_size,
        epochs = args.max_epoch,
        callbacks = [save_checkpoint, reduce_lr])


