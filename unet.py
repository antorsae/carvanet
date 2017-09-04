from keras.models import Model
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose, Add, Dropout, BatchNormalization, Activation, SpatialDropout2D
from keras_contrib.initializers import ConvolutionAware
from keras_contrib.layers.normalization import InstanceNormalization, BatchRenormalization
from keras_contrib.layers.convolutional import CosineConv2D
from keras.initializers import Constant
from pooling import MaxPoolingWithArgmax2D, MaxUnpooling2D

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

def U2Net(input_shape, activation='sigmoid', int_activation='relu', yuv=True):

    SX = input_shape[1]
    SY = input_shape[0]

    assert yuv == True

    Y  = Input(shape=(SY,       SX,     1))
    UV = Input(shape=(SY//2,    SX//2,  2))

    f = 32

    conv1 = Conv2D(f, (3, 3), activation=int_activation, padding='same')(Y)
    conv1 = Conv2D(f, (3, 3), activation=int_activation, padding='same')(conv1)
    pool1 = Conv2D(f, (2, 2), strides=(2,2), activation=int_activation, padding='same')(conv1)
    #pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    convC = Conv2D(f, (3, 3), activation=int_activation, padding='same')(UV)
    convC = Conv2D(f, (3, 3), activation=int_activation, padding='same')(convC)
    pool1_plus_UV = concatenate([pool1, convC], axis=3)

    conv2 = Conv2D(f*2, (3, 3), activation=int_activation, padding='same')(pool1_plus_UV)
    conv2 = Conv2D(f*2, (3, 3), activation=int_activation, padding='same')(conv2)
    pool2 = Conv2D(f*2, (2, 2), strides=(2,2), activation=int_activation, padding='same')(conv2)

#    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(f*4, (3, 3), activation=int_activation, padding='same')(pool2)
    conv3 = Conv2D(f*4, (3, 3), activation=int_activation, padding='same')(conv3)
    pool3 = Conv2D(f*4, (2, 2), strides=(2,2), activation=int_activation, padding='same')(conv3)

    #pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(f*8, (3, 3), activation=int_activation, padding='same')(pool3)
    conv4 = Conv2D(f*8, (3, 3), activation=int_activation, padding='same')(conv4)
    pool4 = Conv2D(f*8, (2, 2), strides=(2,2), activation=int_activation, padding='same')(conv4)

    #pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(f*16, (3, 3), activation=int_activation, padding='same')(pool4)
    conv5 = Conv2D(f*16, (3, 3), activation=int_activation, padding='same')(conv5)

    up6 = concatenate([Conv2DTranspose(f*8, (2, 2), strides=(2, 2), padding='same')(conv5), conv4], axis=3)
    conv6 = Conv2D(f*8, (3, 3), activation=int_activation, padding='same')(up6)
    conv6 = Conv2D(f*8, (3, 3), activation=int_activation, padding='same')(conv6)

    up7 = concatenate([Conv2DTranspose(f*4, (2, 2), strides=(2, 2), padding='same')(conv6), conv3], axis=3)
    conv7 = Conv2D(f*4, (3, 3), activation=int_activation, padding='same')(up7)
    conv7 = Conv2D(f*4, (3, 3), activation=int_activation, padding='same')(conv7)

    up8 = concatenate([Conv2DTranspose(f*2, (2, 2), strides=(2, 2), padding='same')(conv7), conv2], axis=3)
    conv8 = Conv2D(f*2, (3, 3), activation=int_activation, padding='same')(up8)
    conv8 = Conv2D(f*2, (3, 3), activation=int_activation, padding='same')(conv8)

    up9 = concatenate([Conv2DTranspose(f, (2, 2), strides=(2, 2), padding='same')(conv8), conv1], axis=3)
    conv9 = Conv2D(f, (3, 3), activation=int_activation, padding='same')(up9)
    conv9 = Conv2D(f, (3, 3), activation=int_activation, padding='same')(conv9)

    conv10 = Conv2D(1, (1, 1), activation=activation)(conv9)

    model = Model(inputs=[Y, UV], outputs=[conv10])

    return model

def U3Net(input_shape, activation='sigmoid', int_activation='relu', yuv=False, dropout_rate = 0., half=False):

    SX = input_shape[1]
    SY = input_shape[0] // (2 if half else 1 )

    if yuv:
        Y  = Input(shape=(SY,       SX,     1))
        UV = Input(shape=(SY//2,    SX//2,  2))
    else:
        RGB = Input(shape=(SY,       SX,     3))

    int_ini = 'he_normal'
    int_activation = 'relu'

    def C2D(*args,**kwargs):
        return Conv2D(*args, **kwargs)
        C = CosineConv2D
        if 'dilation_rate' in kwargs:
            if kwargs['dilation_rate'] != 1:
                C = Conv2D
            else:
                kwargs.pop('dilation_rate')

        return C(*args, **kwargs)


    def downblock(f, channels, kernel, use_res, batch_norm = True, downsample = True, dilations = None):
        if not dilations:
            dilations = [1] * len(channels)
        features = []
        lc = len(channels)
        #dense_points = set([lc, 3*lc//4, lc//2, lc//4 if lc//4 >2 else 3]) - set([1,2])
        dense_points = range(3,lc+1)

        receptive_field = 1

        for i, (channel, dilation) in enumerate(zip(channels, dilations)):
            if (i+1) not in dense_points:
                ff  = f
                res = f
            else:
                ff = list(features)
                #ff.append(f)
                ff = concatenate(ff, axis=3)
                res = None
#            if (i == len(channels) - 1) and batch_norm:
            if ((i+1) in dense_points) and batch_norm:
                f = C2D(channel, kernel, padding='same', kernel_initializer=int_ini, dilation_rate= dilation)(ff)
                #f = BatchRenormalization(axis=3)(f)
                f = InstanceNormalization(axis=3)(f)
                f = Activation(int_activation)(f)
            else:
                f = C2D(channel, kernel, activation = int_activation, padding='same', kernel_initializer=int_ini, dilation_rate= dilation)(ff)

            if res is not None and (f.shape[-1] == res.shape[-1]) and use_res:
                f = Add()([f, res])
            features.append(f)
            receptive_field += (kernel - 1) * dilation

        f = SpatialDropout2D(0.02)(f)

        if downsample:
#            return (f, MaxPooling2D(pool_size=(2, 2))(f))
            return (f, Conv2D(channels[-1], (2,2), strides=(2,2), activation = int_activation, padding='same', kernel_initializer=int_ini)(f), receptive_field)

#            return (f, MaxPoolingWithArgmax2D(pool_size=(2, 2))(f))
        else:
            return (f, receptive_field)

    def upblock(f, s, channels, kernel, use_res, batch_norm = True, dilations = None, resize_conv=False):
        features = []
        if not dilations:
            dilations = [1] * len(channels)
        if resize_conv:
            u = Conv2DTranspose(channels[0], (2, 2), strides=(2, 2), use_bias=False, padding='same', kernel_initializer=Constant(value=1), trainable=False)(f)
            u = C2D(channels[0], kernel, activation=int_activation, padding='same', kernel_initializer=int_ini)(u)
            u = concatenate([u, s], axis=3)
        else:
            u = concatenate([Conv2DTranspose(channels[0], (2, 2), strides=(2, 2), use_bias=False, padding='same', kernel_initializer=int_ini)(f), s], axis=3)
        lc = len(channels)
        #dense_points = set([lc, 3*lc//4, lc//2, lc//4 if lc//4 >2 else 3]) - set([1,2])
        dense_points = range(3,lc+1)

        for i, (channel, dilation) in enumerate(zip(channels, dilations)):
            print(i)
            if (i+1) not in dense_points:
#            if i < ((len(channels)//2)-1):
                ff  = u
                res = u
            else:
                ff = list(features)
                #ff.append(u)
                ff = concatenate(ff, axis=3)
                res = None
            if ((i+1) in dense_points) and batch_norm:
            #if (i == len(channels) - 1) and batch_norm:
                u = C2D(channel, kernel, padding='same', kernel_initializer=int_ini, dilation_rate=dilation)(ff)
                #u = BatchRenormalization(axis=3)(u)
                u = InstanceNormalization(axis=3)(u)
                u = Activation(int_activation)(u)
            else:
                u = C2D(channel, kernel, activation=int_activation, padding='same', kernel_initializer=int_ini, dilation_rate= dilation)(ff)
            if res is not None and (u.shape[-1] == res.shape[-1]) and use_res:
                u = Add()([u, res])
            features.append(u)
        return u, features

    skip = []
    K = 3
    J = 3

    g   = 12*2

    gY  = 12
    gUV = 8

    l   = 4
    lY  = 8
    lU  = 6

    cY,c, receptive_field  = downblock(Y, channels = [gY] * lY, kernel = J, use_res = False)
    skip.append(cY)

    cUV, _ = downblock(UV, channels = [gUV] * lU, kernel = K, use_res = False, downsample=False)

    c = concatenate([c, cUV], axis=3)

    downblocks = [[g*4//3] * l, [g*2] * 3]#,  [g] * l]
    dilations  = [[1] * l,      [2, 4, 8] ]#,  [2] * l]
    res        = [False,    False]

    for i, (db, di, ds) in enumerate(zip(downblocks, dilations, res)):
        cs,c, rf = downblock(c, channels = db , kernel = K, use_res = ds, dilations = di)
        receptive_field += rf * i 
        skip.append(cs)

    c, rf = downblock(c, channels = downblocks[-1], kernel = K, use_res = res[-1], dilations = dilations[-1], downsample=False)
    receptive_field += rf*i

    print("Receptive field: " + str(receptive_field) + " pixels")


    upblocks = list(downblocks[::-1])
    upblocks.append([gY] * lY)

    dilations = dilations[::-1]
    dilations.append([1] * lY)

    res       = res[::-1]
    res.append(False)

    print(upblocks)

    for l, (ub, ui, us) in enumerate(zip(upblocks, dilations, res)):
        c, cc = upblock(c, skip[-l-1], channels = ub , kernel = K if l != (len(upblocks)-1) else J, use_res = us, dilations = ui)

    c = Conv2D(1, (1, 1), activation=activation, kernel_initializer='he_uniform')(concatenate(cc, axis=3))

    model = Model(inputs=[Y, UV], outputs=[c])

    return model

def RNet(input_shape, activation='sigmoid', int_activation='relu', yuv=True, dropout_rate=0.):

    SX = input_shape[1]
    SY = input_shape[0]

    assert yuv == True

    Y  = Input(shape=(SY,       SX,     1))
    UV = Input(shape=(SY//2,    SX//2,  2))

    fY  = 8
    fUV = 2
    f = fY + fUV
    K = 7

    convY  = Conv2D(fY, K, activation=int_activation, padding='same')(Y)
    convUV = Conv2DTranspose(fUV, K, strides=(2, 2), padding='same')(UV)
    conv1  = concatenate([convY, convUV], axis=3)

    def block(conv, residual, f, dilation):
        conv  = Conv2D(f, K, activation=int_activation, padding='same', dilation_rate=dilation)(conv)
        if residual:
            res    = conv
        conv  = Conv2D(f, K, activation=int_activation, padding='same', dilation_rate=dilation)(conv)
        if residual:
            conv  = Add()([conv, res])
        return conv

    conv2 = block(conv1, True,  f,   1)
    #conv3 = concatenate([conv1, conv2], axis=3)
    conv3 = block(conv2, True,  f,   2)
    #conv3 = concatenate([conv1, conv2, conv3], axis=3)
    conv4 = block(conv3, True,  f,   4)
    #conv3 = concatenate([conv1, conv2, conv3, conv4], axis=3)
    conv5 = block(conv4, True,  f,   8)

    conv6 = block(conv5, True,   f , 8)
    #conv6 = concatenate([conv5, conv6], axis=3)
    conv7 = block(conv6, False,  f,  4)
    #conv7 = concatenate([conv5, conv6, conv7], axis=3)
    conv8 = block(conv7, False,  f , 4)
    #conv8 = concatenate([conv5, conv6, conv7, conv8], axis=3)
    conv9 = block(conv8, False,  f , 2)
    #conv9 = concatenate([conv5, conv6, conv7, conv8, conv9], axis=3)
    conv10 = block(conv9, False, f,  1)
    conv10 = concatenate([conv1, conv2, conv3, conv4, conv5, conv6, conv7, conv8, conv9, conv10], axis=3)
    conv11 = block(conv10, True,  f*8 , 1)

    seg = Conv2D(1, (1, 1), activation=activation)(conv11)

    model = Model(inputs=[Y, UV], outputs=[seg])

    return model
