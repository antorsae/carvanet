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


import densenet_fc as dc

parser = argparse.ArgumentParser()
parser.add_argument('--max-epoch', type=int, default=200, help='Epoch to run')
parser.add_argument('-b', '--batch-size', type=int, default=1, help='Batch Size during training, e.g. -b 2')
parser.add_argument('-l', '--learning_rate', type=float, default=1e-3, help='Initial learning rate')
args = parser.parse_args()

SX = 1920
SY = 1280

def gen(items, batch_size, training=True):
    X = np.zeros((batch_size, SY, SX, 3), dtype=np.float32)
    y = np.zeros((batch_size, SY, SX, 1), dtype=np.float32)
    
    assert batch_size <= 16
    input_folder = '.'

    load_img   = lambda im, idx: imread(join(input_folder, 'train', '{}_{:02d}.jpg'.format(im, idx)))
    load_mask  = lambda im, idx: imread(join(input_folder, 'train_masks', '{}_{:02d}_mask.gif'.format(im, idx)))
    mask_image = lambda im, mask: (im * np.expand_dims(mask, 2))

    imgs_idx   = list(range(1, 17))

    batch_idx = 0

    while True:
        if training:
            random.shuffle(items)

        for item in items:

            for idx in imgs_idx:

                X[batch_idx, :, :1918, :] = load_img(item, idx) / 255.
                X[batch_idx, :, 1918:, :] = 0.

                y[batch_idx, :, :1918, :] = load_mask(item, idx)[...,:1] / 255.
                y[batch_idx, :, 1918:, :] = 0.

                #print(np.amin(X[batch_idx]), np.amax(X[batch_idx]))
                #print(np.amin(y[batch_idx]), np.amax(y[batch_idx]))

                batch_idx += 1

                if batch_idx == batch_size:
                    X -= 0.5
                    yield(X, y)
                    batch_idx = 0


ids = list(set([(x.split('_')[0]).split('/')[1] for x in glob.glob('train/*_*.jpg')]))

ids_train, ids_val = train_test_split(ids, test_size=0.2, random_state=42)

model = dc.DenseNetFCN((SY, SX, 3), nb_dense_block=5, growth_rate=6, 
	nb_layers_per_block=4, upsampling_type='upsampling', classes=1, activation='sigmoid', 
	batch_norm=False, int_activation='relu')

model.summary()

smooth = 1.

# From here: https://github.com/jocicmarko/ultrasound-nerve-segmentation/blob/master/train.py
def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def bce_dice_loss(y_true, y_pred):
    return 0.5 * binary_crossentropy(y_true, y_pred) - dice_coef(y_true, y_pred)

model.compile(Adam(lr=args.learning_rate), bce_dice_loss, metrics=['accuracy', dice_coef])

model.fit_generator(
        generator        = gen(ids_train, args.batch_size),
        steps_per_epoch  = len(ids_train) * 16 // args.batch_size,
        validation_data  = gen(ids_val, args.batch_size, training = False),
        validation_steps = len(ids_val) * 16 // args.batch_size,
        epochs = args.max_epoch,
        callbacks = [])


