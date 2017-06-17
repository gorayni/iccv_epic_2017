from __future__ import division

import numpy as np

from keras import backend as K
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
from experiments.utils import HistoryLog
from keras.layers import Dense
from keras.layers import GlobalAveragePooling2D
from keras.applications.inception_v3 import InceptionV3
from keras.models import Model
from keras.layers import Dropout
import experiments as exp
from keras.preprocessing import image

import os
import datetime
import time


from keras.models import Model
from keras.layers import Flatten, Dense, Input, BatchNormalization, merge
from keras.layers import Convolution2D, MaxPooling2D, AveragePooling2D
from keras.layers import Dropout
from keras.utils.layer_utils import convert_all_kernels_in_model
from keras.utils.data_utils import get_file
from keras import backend as K


def get_session(gpu_fraction=0.8):
    import tensorflow as tf

    num_threads = os.environ.get('OMP_NUM_THREADS')
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)

    if num_threads:
        return tf.Session(config=tf.ConfigProto(
            gpu_options=gpu_options, intra_op_parallelism_threads=num_threads))
    else:
        return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))


def train_net():
    img_width, img_height = 299, 299

    seed = 42
    np.random.seed(seed)

    base_model = InceptionV3(include_top=False, weights='imagenet')
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.5)(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(21, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)

    for layer in model.layers[:249]:
       layer.trainable = False
    for layer in model.layers[249:]:
       layer.trainable = True

    model.summary()
    sgd = SGD(lr=0.0001, momentum=0.9)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    # prepare data augmentation configuration
    train_datagen = ImageDataGenerator(rescale=1. / 255,
                                       rotation_range=40,
                                       width_shift_range=0.2,
                                       height_shift_range=0.2,
                                       zoom_range=0.2,
                                       horizontal_flip=True)

    val_datagen = ImageDataGenerator(rescale=1. / 255)

    train_generator = train_datagen.flow_from_directory('data/training',
                                                        target_size=(img_height, img_width),
                                                        class_mode='categorical',
                                                        batch_size=1)

    validation_generator = val_datagen.flow_from_directory(
        'data/validation',
        target_size=(img_height, img_width),
        class_mode='categorical',
        batch_size=1)

    # checkpoint
    weights_filepath = "weights.InceptionV3.{epoch:02d}.hdf5"
    checkpoint = ModelCheckpoint(weights_filepath, monitor='val_acc', verbose=1)
    history = HistoryLog()

    # fine-tune the model
    model.fit_generator(
        train_generator,
        steps_per_epoch=36095,# 36095,#15,
        epochs=10,
        callbacks=[checkpoint, history],
        validation_data=validation_generator,
        validation_steps=6225)  # 6225)#20)

    ts = time.time()
    st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d_%H:%M:%S')

    loss_filepath = "InceptionV3.first_phase.loss." + st + ".log"
    history.log_training_loss(loss_filepath)

    epoch_filepath = "InceptionV3.first_phase.epoch." + st + ".log"
    history.log_epoch(epoch_filepath)

    if K.backend() == 'tensorflow':
        K.clear_session()


if __name__ == '__main__':

    if K.backend() == 'tensorflow':
        import keras.backend.tensorflow_backend as KTF

        KTF.set_session(get_session(0.8))
    train_net()
