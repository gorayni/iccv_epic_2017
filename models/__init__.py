from __future__ import division
from keras.applications import vgg16
from keras.layers import Dense, Flatten
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
        model.load_weights('weights.VGG-16.FC.09.hdf5')
    return model
