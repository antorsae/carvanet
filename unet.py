from keras.models import Model
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose

def UNet(input_shape, nb_dense_block=5, growth_rate=16, nb_layers_per_block=4,
                reduction=0.0, dropout_rate=0.0, weight_decay=1E-4, init_conv_filters=48,
                include_top=True, weights=None, input_tensor=None, classes=1, activation='sigmoid',
                upsampling_conv=128, upsampling_type='upsampling', batchsize=None, batch_norm=True, int_activation='relu'):

    inputs = Input(input_shape)
    conv1 = Conv2D(32, (3, 3), activation=int_activation, padding='same')(inputs)
    conv1 = Conv2D(32, (3, 3), activation=int_activation, padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(64, (3, 3), activation=int_activation, padding='same')(pool1)
    conv2 = Conv2D(64, (3, 3), activation=int_activation, padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(128, (3, 3), activation=int_activation, padding='same')(pool2)
    conv3 = Conv2D(128, (3, 3), activation=int_activation, padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(256, (3, 3), activation=int_activation, padding='same')(pool3)
    conv4 = Conv2D(256, (3, 3), activation=int_activation, padding='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(512, (3, 3), activation=int_activation, padding='same')(pool4)
    conv5 = Conv2D(512, (3, 3), activation=int_activation, padding='same')(conv5)

    up6 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv5), conv4], axis=3)
    conv6 = Conv2D(256, (3, 3), activation=int_activation, padding='same')(up6)
    conv6 = Conv2D(256, (3, 3), activation=int_activation, padding='same')(conv6)

    up7 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv6), conv3], axis=3)
    conv7 = Conv2D(128, (3, 3), activation=int_activation, padding='same')(up7)
    conv7 = Conv2D(128, (3, 3), activation=int_activation, padding='same')(conv7)

    up8 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv7), conv2], axis=3)
    conv8 = Conv2D(64, (3, 3), activation=int_activation, padding='same')(up8)
    conv8 = Conv2D(64, (3, 3), activation=int_activation, padding='same')(conv8)

    up9 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv8), conv1], axis=3)
    conv9 = Conv2D(32, (3, 3), activation=int_activation, padding='same')(up9)
    conv9 = Conv2D(32, (3, 3), activation=int_activation, padding='same')(conv9)

    conv10 = Conv2D(1, (1, 1), activation=activation)(conv9)

    model = Model(inputs=[inputs], outputs=[conv10])

    return model

def U2Net(input_shape, activation='sigmoid', int_activation='relu'):

    SX = input_shape[1]
    SY = input_shape[0]
    assert input_shape[2] == 3
    Y  = Input(shape=(SY,       SX,     1))
    UV = Input(shape=(SY//2,    SX//2,  2))

    conv1 = Conv2D(32, (3, 3), activation=int_activation, padding='same')(Y)
    conv1 = Conv2D(32, (3, 3), activation=int_activation, padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    convC = Conv2D(32, (3, 3), activation=int_activation, padding='same')(UV)
    convC = Conv2D(32, (3, 3), activation=int_activation, padding='same')(convC)
    pool1_plus_UV = concatenate([pool1, convC], axis=3)

    conv2 = Conv2D(64, (3, 3), activation=int_activation, padding='same')(pool1_plus_UV)
    conv2 = Conv2D(64, (3, 3), activation=int_activation, padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(128, (3, 3), activation=int_activation, padding='same')(pool2)
    conv3 = Conv2D(128, (3, 3), activation=int_activation, padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(256, (3, 3), activation=int_activation, padding='same')(pool3)
    conv4 = Conv2D(256, (3, 3), activation=int_activation, padding='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(512, (3, 3), activation=int_activation, padding='same')(pool4)
    conv5 = Conv2D(512, (3, 3), activation=int_activation, padding='same')(conv5)

    up6 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv5), conv4], axis=3)
    conv6 = Conv2D(256, (3, 3), activation=int_activation, padding='same')(up6)
    conv6 = Conv2D(256, (3, 3), activation=int_activation, padding='same')(conv6)

    up7 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv6), conv3], axis=3)
    conv7 = Conv2D(128, (3, 3), activation=int_activation, padding='same')(up7)
    conv7 = Conv2D(128, (3, 3), activation=int_activation, padding='same')(conv7)

    up8 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv7), conv2], axis=3)
    conv8 = Conv2D(64, (3, 3), activation=int_activation, padding='same')(up8)
    conv8 = Conv2D(64, (3, 3), activation=int_activation, padding='same')(conv8)

    up9 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv8), conv1], axis=3)
    conv9 = Conv2D(32, (3, 3), activation=int_activation, padding='same')(up9)
    conv9 = Conv2D(32, (3, 3), activation=int_activation, padding='same')(conv9)

    conv10 = Conv2D(1, (1, 1), activation=activation)(conv9)

    model = Model(inputs=[Y, UV], outputs=[conv10])

    return model