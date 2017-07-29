import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.transform import resize
from os.path import join
from tqdm import tqdm_notebook
import cv2
from sklearn.model_selection import train_test_split

import densenet_fc as dc

input_folder = '.'

df_mask    = pd.read_csv(join(input_folder, 'train_masks.csv'), usecols=['img'])
ids_train  = df_mask['img'].map(lambda s: s.split('_')[0]).unique()

imgs_idx   = list(range(1, 17))

SX = 1920//2
SY = 1280//2

load_img   = lambda im, idx: imread(join(input_folder, 'train', '{}_{:02d}.jpg'.format(im, idx)))
load_mask  = lambda im, idx: imread(join(input_folder, 'train_masks', '{}_{:02d}_mask.gif'.format(im, idx)))
downsize   = lambda im: resize(im, (SY, SX))
mask_image = lambda im, mask: (im * np.expand_dims(mask, 2))

num_train = len(ids_train)

# Load data for position id=1
X = np.empty((num_train, SY, SX, 12), dtype=np.float32)
y = np.empty((num_train, SY, SX, 1),  dtype=np.float32)

idx = 1 # Rotation index
for i, img_id in enumerate(ids_train[:num_train]):
    imgs_id = [downsize(load_img(img_id, j)) for j in imgs_idx]
    # Input is image + mean image per channel + std image per channel
    X[i, ..., :9] = np.concatenate([imgs_id[idx-1], np.mean(imgs_id, axis=0), np.std(imgs_id, axis=0)], axis=2)
    y[i] = downsize(np.expand_dims(load_mask(img_id, idx), 2)) / 255.
    del imgs_id # Free memory

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Concat overall y info to X
# This is important as the kernels of CNN used below has no information of its location
y_train_mean = y_train.mean(axis=0)
y_train_std = y_train.std(axis=0)
y_train_min = y_train.min(axis=0)

y_features = np.concatenate([y_train_mean, y_train_std, y_train_min], axis=2)

X_train[:, ..., -3:] = y_features
X_val[:, ..., -3:] = y_features

# Normalize input and output
X_mean = X_train.mean(axis=(0,1,2), keepdims=True)
X_std = X_train.std(axis=(0,1,2), keepdims=True)

X_train -= X_mean
X_train /= X_std

X_val -= X_mean
X_val /= X_std

# Create simple model
from keras.layers import Conv2D
from keras.models import Sequential
import keras.backend as K

model = dc.DenseNetFCN((SY, SX, 12), nb_dense_block=5, growth_rate=16, 
	nb_layers_per_block=4, upsampling_type='upsampling', classes=1, activation='sigmoid', 
	batch_norm=False, int_activation='relu')

model.summary()


#model = Sequential()
#model.add( Conv2D(16, 3, activation='relu', padding='same', input_shape=(320, 480, 12) ) )
#model.add( Conv2D(32, 3, activation='relu', padding='same') )
#model.add( Conv2D(1, 5, activation='sigmoid', padding='same') )


from keras.optimizers import Adam
from keras.losses import binary_crossentropy

smooth = 1.

# From here: https://github.com/jocicmarko/ultrasound-nerve-segmentation/blob/master/train.py
def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def bce_dice_loss(y_true, y_pred):
    return 0.5 * binary_crossentropy(y_true, y_pred) - dice_coef(y_true, y_pred)

model.compile(Adam(lr=1e-3), bce_dice_loss, metrics=['accuracy', dice_coef])

history = model.fit(X_train, y_train, epochs=200, validation_data=(X_val, y_val), batch_size=1, verbose=2)


