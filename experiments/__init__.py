from __future__ import division
from keras.applications import vgg16
from keras.models import Sequential
from keras.layers.wrappers import TimeDistributed
from keras.layers import Dense, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers.recurrent import LSTM
from keras.constraints import maxnorm
from keras.models import Model
from keras.layers import Dropout


def vgg16_first_phase_model(weights='imagenet', img_width=224, img_height=224):
    base_model = vgg16.VGG16(include_top=False, weights=weights, input_shape=(img_width, img_height, 3))

    for layer in base_model.layers:
        layer.trainable = False

    x = base_model.output
    x = Flatten(name='flatten')(x)
    x = Dense(4096, activation='relu', name='fc1', W_constraint=maxnorm(3))(x)
    x = Dropout(0.5)(x)
    x = Dense(4096, activation='relu', name='fc2', W_constraint=maxnorm(3))(x)
    x = Dropout(0.5)(x)
    x = Dense(21, activation='softmax', name='predictions')(x)

    return Model(input=base_model.input, output=x)


def vgg_16_second_phase_model(weights=None, img_width=224, img_height=224):
    model = vgg16_first_phase_model(None, img_width, img_height)

    for layer in model.layers[15:19]:
        layer.trainable = True

    if weights:
        model.load_weights(weights)
    return model


def vgg_16_plus_lstm(weights=None, vgg16_weights=None, timestep=10, img_width=224, img_height=224):
    input_shape = (timestep, img_width, img_height, 3)

    # Block 1
    model = Sequential(name='vgg16+lstm');

    model.add(TimeDistributed(Convolution2D(64, 3, 3, activation='relu', border_mode='same'), name='block1_conv1',
                              input_shape=input_shape))
    model.add(TimeDistributed(Convolution2D(64, 3, 3, activation='relu', border_mode='same'), name='block1_conv2'))
    model.add(TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2)), name='block1_pool'))

    # Block 2
    model.add(TimeDistributed(Convolution2D(128, 3, 3, activation='relu', border_mode='same'), name='block2_conv1'))
    model.add(TimeDistributed(Convolution2D(128, 3, 3, activation='relu', border_mode='same'), name='block2_conv2'))
    model.add(TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2)), name='block2_pool'))

    # Block 3
    model.add(TimeDistributed(Convolution2D(256, 3, 3, activation='relu', border_mode='same'), name='block3_conv1'))
    model.add(TimeDistributed(Convolution2D(256, 3, 3, activation='relu', border_mode='same'), name='block3_conv2'))
    model.add(TimeDistributed(Convolution2D(256, 3, 3, activation='relu', border_mode='same'), name='block3_conv3'))
    model.add(TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2)), name='block3_pool'))

    # Block 4
    model.add(TimeDistributed(Convolution2D(512, 3, 3, activation='relu', border_mode='same'), name='block4_conv1'))
    model.add(TimeDistributed(Convolution2D(512, 3, 3, activation='relu', border_mode='same'), name='block4_conv2'))
    model.add(TimeDistributed(Convolution2D(512, 3, 3, activation='relu', border_mode='same'), name='block4_conv3'))
    model.add(TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2)), name='block4_pool'))

    for layer in model.layers:
        layer.trainable = False

    # # Block 5
    model.add(TimeDistributed(Convolution2D(512, 3, 3, activation='relu', border_mode='same'), name='block5_conv1'))
    model.add(TimeDistributed(Convolution2D(512, 3, 3, activation='relu', border_mode='same'), name='block5_conv2'))
    model.add(TimeDistributed(Convolution2D(512, 3, 3, activation='relu', border_mode='same'), name='block5_conv3'))
    model.add(TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2)), name='block5_pool'))

    # Classification block
    model.add(TimeDistributed(Flatten(), name='flatten'))
    model.add(TimeDistributed(Dense(4096, activation='relu', W_constraint=maxnorm(3)), name='fc1'))

    if vgg16_weights and not weights:
        model.load_weights(vgg16_weights, by_name=True)

    model.add(TimeDistributed(Dropout(0.5)))
    model.add(LSTM(512, name='lstm1', return_sequences=True))
    model.add(TimeDistributed(Dropout(0.5)))
    model.add(TimeDistributed(Dense(21, activation='softmax'), name='predictions'))

    if weights:
        model.load_weights(weights)
    return model
