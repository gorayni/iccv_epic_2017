from __future__ import division
from keras.applications import vgg16
from keras.applications.inception_v3 import InceptionV3
from keras.models import Sequential
from keras.layers.wrappers import TimeDistributed
from keras.layers import Dense, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.constraints import maxnorm
from keras.models import Model
from keras.layers import Dropout
from keras.layers import Input
from keras.layers import LSTM
from keras.layers import add
from keras.layers import multiply
from keras.layers import GlobalAveragePooling2D


def inceptionV3_first_phase_model(weights='imagenet', img_width=299, img_height=299):
    base_model = InceptionV3(weights=weights, include_top=False, input_shape=(img_width, img_height, 3))

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(21, activation='softmax')(x)

    # this is the model we will train
    model = Model(inputs=base_model.input, outputs=predictions)

    for layer in base_model.layers:
        layer.trainable = False

    return model


def vgg16_first_phase_model(weights='imagenet', img_width=224, img_height=224):
    base_model = vgg16.VGG16(include_top=False, weights=weights, input_shape=(img_width, img_height, 3))

    for layer in base_model.layers:
        layer.trainable = False

    x = base_model.output
    x = Flatten(name='flatten')(x)
    x = Dense(4096, activation='relu', name='fc1', kernel_constraint=maxnorm(2.))(x)
    x = Dropout(0.5)(x)
    x = Dense(4096, activation='relu', name='fc2', kernel_constraint=maxnorm(2.))(x)
    x = Dropout(0.5)(x)
    x = Dense(21, activation='softmax', name='predictions')(x)

    return Model(name='VGG-16', inputs=[base_model.input], outputs=[x])


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

    model.add(TimeDistributed(Convolution2D(64, (3, 3), activation='relu', padding='same', trainable=False),
                              name='block1_conv1',
                              input_shape=input_shape))
    model.add(TimeDistributed(Convolution2D(64, (3, 3), activation='relu', padding='same', trainable=False),
                              name='block1_conv2'))
    model.add(TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2)), name='block1_pool'))

    # Block 2
    model.add(TimeDistributed(Convolution2D(128, (3, 3), activation='relu', padding='same', trainable=False),
                              name='block2_conv1'))
    model.add(TimeDistributed(Convolution2D(128, (3, 3), activation='relu', padding='same', trainable=False),
                              name='block2_conv2'))
    model.add(TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2)), name='block2_pool'))

    # Block 3
    model.add(TimeDistributed(Convolution2D(256, (3, 3), activation='relu', padding='same', trainable=False),
                              name='block3_conv1'))
    model.add(TimeDistributed(Convolution2D(256, (3, 3), activation='relu', padding='same', trainable=False),
                              name='block3_conv2'))
    model.add(TimeDistributed(Convolution2D(256, (3, 3), activation='relu', padding='same', trainable=False),
                              name='block3_conv3'))
    model.add(TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2)), name='block3_pool'))

    # Block 4
    model.add(TimeDistributed(Convolution2D(512, (3, 3), activation='relu', padding='same', trainable=False),
                              name='block4_conv1'))
    model.add(TimeDistributed(Convolution2D(512, (3, 3), activation='relu', padding='same', trainable=False),
                              name='block4_conv2'))
    model.add(TimeDistributed(Convolution2D(512, (3, 3), activation='relu', padding='same', trainable=False),
                              name='block4_conv3'))
    model.add(TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2)), name='block4_pool'))

    # Block 5
    model.add(TimeDistributed(Convolution2D(512, (3, 3), activation='relu', padding='same'), name='block5_conv1'))
    model.add(TimeDistributed(Convolution2D(512, (3, 3), activation='relu', padding='same'), name='block5_conv2'))
    model.add(TimeDistributed(Convolution2D(512, (3, 3), activation='relu', padding='same'), name='block5_conv3'))
    model.add(TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2)), name='block5_pool'))

    # Classification block
    model.add(TimeDistributed(Flatten(), name='flatten'))
    model.add(TimeDistributed(Dense(4096, activation='relu', kernel_constraint=maxnorm(2.)), name='fc1'))

    if vgg16_weights and not weights:
        model.load_weights(vgg16_weights, by_name=True)

    model.add(TimeDistributed(Dropout(0.5)))
    model.add(LSTM(256, name='lstm1', return_sequences=True))
    model.add(TimeDistributed(Dropout(0.5)))
    model.add(TimeDistributed(Dense(21, activation='softmax'), name='predictions'))

    if weights:
        model.load_weights(weights)
    return model


def filtered_vgg_16_plus_lstm(weights=None, timestep=10, img_width=224, img_height=224):
    input_shape = (timestep, img_width, img_height, 3)

    main_input = Input(shape=input_shape)
    mask_input = Input(shape=(timestep, 21))
    prev_values_input = Input(shape=(timestep, 21))

    # Block 1
    x = TimeDistributed(Convolution2D(64, (3, 3), activation='relu', padding='same', trainable=False),
                        name='block1_conv1')(main_input)
    x = TimeDistributed(Convolution2D(64, (3, 3), activation='relu', padding='same', trainable=False),
                        name='block1_conv2')(x)
    x = TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2)), name='block1_pool')(x)

    # Block 2
    x = TimeDistributed(Convolution2D(128, (3, 3), activation='relu', padding='same', trainable=False),
                        name='block2_conv1')(x)
    x = TimeDistributed(Convolution2D(128, (3, 3), activation='relu', padding='same', trainable=False),
                        name='block2_conv2')(x)
    x = TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2)), name='block2_pool')(x)

    # # Block 3
    x = TimeDistributed(Convolution2D(256, (3, 3), activation='relu', padding='same', trainable=False),
                        name='block3_conv1')(x)
    x = TimeDistributed(Convolution2D(256, (3, 3), activation='relu', padding='same', trainable=False),
                        name='block3_conv2')(x)
    x = TimeDistributed(Convolution2D(256, (3, 3), activation='relu', padding='same', trainable=False),
                        name='block3_conv3')(x)
    x = TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2)), name='block3_pool')(x)

    # Block 4
    x = TimeDistributed(Convolution2D(512, (3, 3), activation='relu', padding='same', trainable=False),
                        name='block4_conv1')(x)
    x = TimeDistributed(Convolution2D(512, (3, 3), activation='relu', padding='same', trainable=False),
                        name='block4_conv2')(x)
    x = TimeDistributed(Convolution2D(512, (3, 3), activation='relu', padding='same', trainable=False),
                        name='block4_conv3')(x)
    x = TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2)), name='block4_pool')(x)

    # Block 5
    x = TimeDistributed(Convolution2D(512, (3, 3), activation='relu', padding='same'), name='block5_conv1')(x)
    x = TimeDistributed(Convolution2D(512, (3, 3), activation='relu', padding='same'), name='block5_conv2')(x)
    x = TimeDistributed(Convolution2D(512, (3, 3), activation='relu', padding='same'), name='block5_conv3')(x)
    x = TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2)), name='block5_pool')(x)

    # Classification block
    x = TimeDistributed(Flatten(), name='flatten')(x)
    x = TimeDistributed(Dense(4096, activation='relu', kernel_constraint=maxnorm(2.)), name='fc1')(x)
    x = Dropout(0.5)(x)
    x = TimeDistributed(Dense(4096, activation='relu', kernel_constraint=maxnorm(2.)), name='fc2')(x)
    x = Dropout(0.5)(x)
    x = TimeDistributed(Dense(21, activation='softmax'), name='predictions')(x)

    # Filter block
    x = multiply([x, mask_input])
    x = add([x, prev_values_input])

    x = TimeDistributed(Dropout(0.5))(x)
    x = LSTM(64, return_sequences=True, name='lstm1')(x)
    x = TimeDistributed(Dropout(0.5))(x)
    x = TimeDistributed(Dense(21, activation='softmax'), name='predictions_')(x)

    model = Model(name='Filtered_VGG-16+LSTM', inputs=[main_input, mask_input, prev_values_input], outputs=[x])

    if weights:
        model.load_weights(weights, by_name=True)

    return model


def filtered_vgg_16_plus_lstm_first_phase(weights=None, timestep=10, img_width=224, img_height=224):
    input_shape = (timestep, img_width, img_height, 3)

    main_input = Input(shape=input_shape)
    mask_input = Input(shape=(timestep, 21))
    prev_values_input = Input(shape=(timestep, 21))

    # Block 1
    x = TimeDistributed(Convolution2D(64, (3, 3), activation='relu', padding='same', trainable=False),
                        name='block1_conv1')(main_input)
    x = TimeDistributed(Convolution2D(64, (3, 3), activation='relu', padding='same', trainable=False),
                        name='block1_conv2')(x)
    x = TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2)), name='block1_pool')(x)

    # Block 2
    x = TimeDistributed(Convolution2D(128, (3, 3), activation='relu', padding='same', trainable=False),
                        name='block2_conv1')(x)
    x = TimeDistributed(Convolution2D(128, (3, 3), activation='relu', padding='same', trainable=False),
                        name='block2_conv2')(x)
    x = TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2)), name='block2_pool')(x)

    # # Block 3
    x = TimeDistributed(Convolution2D(256, (3, 3), activation='relu', padding='same', trainable=False),
                        name='block3_conv1')(x)
    x = TimeDistributed(Convolution2D(256, (3, 3), activation='relu', padding='same', trainable=False),
                        name='block3_conv2')(x)
    x = TimeDistributed(Convolution2D(256, (3, 3), activation='relu', padding='same', trainable=False),
                        name='block3_conv3')(x)
    x = TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2)), name='block3_pool')(x)

    # Block 4
    x = TimeDistributed(Convolution2D(512, (3, 3), activation='relu', padding='same', trainable=False),
                        name='block4_conv1')(x)
    x = TimeDistributed(Convolution2D(512, (3, 3), activation='relu', padding='same', trainable=False),
                        name='block4_conv2')(x)
    x = TimeDistributed(Convolution2D(512, (3, 3), activation='relu', padding='same', trainable=False),
                        name='block4_conv3')(x)
    x = TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2)), name='block4_pool')(x)

    # Block 5
    x = TimeDistributed(Convolution2D(512, (3, 3), activation='relu', padding='same'), name='block5_conv1')(x)
    x = TimeDistributed(Convolution2D(512, (3, 3), activation='relu', padding='same'), name='block5_conv2')(x)
    x = TimeDistributed(Convolution2D(512, (3, 3), activation='relu', padding='same'), name='block5_conv3')(x)
    x = TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2)), name='block5_pool')(x)

    # Classification block
    x = TimeDistributed(Flatten(), name='flatten')(x)
    x = TimeDistributed(Dense(4096, activation='relu', kernel_constraint=maxnorm(2.)), name='fc1')(x)
    x = TimeDistributed(Dropout(0.5))(x)
    x = TimeDistributed(Dense(256, activation='relu'), name='fc2_')(x)
    x = TimeDistributed(Dropout(0.5))(x)

    # Filter block
    x = multiply([x, mask_input])
    x = add([x, prev_values_input])

    x = LSTM(256, return_sequences=True, name='lstm1')(x)
    x = TimeDistributed(Dropout(0.5))(x)
    x = TimeDistributed(Dense(21, activation='softmax'), name='predictions_')(x)

    model = Model(name='Filtered_VGG-16+LSTM', inputs=[main_input, mask_input, prev_values_input], outputs=[x])

    if weights:
        model.load_weights(weights, by_name=True)

    return model
