from __future__ import division

import numpy as np

from keras import backend as K
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
from experiments.utils import HistoryLog
import experiments as exp

import os


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
    img_width, img_height = 224, 224

    seed = 42
    np.random.seed(seed)

    sgd = SGD(lr=0.000067, decay=0.005, momentum=0.9, nesterov=True)
    model = exp.inceptionV3_first_phase_model()
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
    checkpoint = ModelCheckpoint(weights_filepath, monitor='val_acc', verbose=1, save_best_only=False)
    history = HistoryLog()

    # fine-tune the model
    model.fit_generator(
        train_generator,
        steps_per_epoch=36095,  # 36095,#15,
        epochs=10,
        callbacks=[checkpoint, history],
        validation_data=validation_generator,
        validation_steps=6225)  # 6225)#20)

    loss_filepath = "InceptionV3.first_phase,loss.log"
    history.log_training_loss(loss_filepath)

    epoch_filepath = "InceptionV3.first_phase,epoch.log"
    history.log_epoch(epoch_filepath)

    if K.backend() == 'tensorflow':
        K.clear_session()


if __name__ == '__main__':

    if K.backend() == 'tensorflow':
        import keras.backend.tensorflow_backend as KTF
        KTF.set_session(get_session(0.8))
    train_net()
