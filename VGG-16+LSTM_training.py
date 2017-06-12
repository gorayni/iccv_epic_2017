from __future__ import division

import numpy as np
import ntcir
import ntcir.IO as IO

import argparse

from keras import backend as K
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
from experiments.utils import HistoryLog
from experiments.utils import generate_batch
import experiments as exp
import os

import keras.backend.tensorflow_backend as KTF

def get_session(gpu_fraction=0.8):
    import tensorflow as tf

    num_threads = os.environ.get('OMP_NUM_THREADS')
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)

    if num_threads:
        return tf.Session(config=tf.ConfigProto(
            gpu_options=gpu_options, intra_op_parallelism_threads=num_threads))
    else:
        return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

def train_net(timestep=10):
    users = IO.load_annotations(ntcir.filepaths)
    sorted_users = ntcir.utils.sort(users)

    # Full day sequences
    num_frames_per_day = 2880
    sequences = ntcir.get_sequences(sorted_users, num_frames_per_day)

    training_set = ntcir.read_split('training_split.txt')
    validation_set = ntcir.read_split('validation_split.txt')

    training_batches = ntcir.get_training_batches(training_set, sequences, batch_size=timestep)
    validation_batches = ntcir.get_batches(validation_set, sequences, batch_size=timestep)

    backend = 'tf' if K.backend() == 'tensorflow' else 'th'

    base_model_weights = 'weights.VGG-16.best.{}.hdf5'.format(backend)
    for learning_rate in [0.0001, 0.00005, 0.000025, 0.000075]:
        K.set_learning_phase(1)

        np.random.seed(42)
        sgd = SGD(lr=learning_rate, decay=0.000005, momentum=0.9, nesterov=True)

        model = exp.vgg_16_plus_lstm(vgg16_weights=base_model_weights, timestep=timestep)
        model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

        # prepare data augmentation configuration
        train_datagen = ImageDataGenerator(rescale=1. / 255,
                                           rotation_range=40,
                                           width_shift_range=0.2,
                                           height_shift_range=0.2,
                                           zoom_range=0.2,
                                           horizontal_flip=True)

        val_datagen = ImageDataGenerator(rescale=1. / 255)

        train_generator = generate_batch(train_datagen, users, training_batches)

        val_generator = generate_batch(val_datagen, users, validation_batches)

        # checkpoint
        weights_filepath = "weights.VGG-16+LSTM.timesteps_" + str(timestep) + ".lr_" + str(
            learning_rate) + ".{epoch:02d}." + backend + ".hdf5"
        checkpoint = ModelCheckpoint(weights_filepath, monitor='val_acc', verbose=1)
        history = HistoryLog()

        # fine-tune the model
        model.fit_generator(
            train_generator,
            steps_per_epoch=len(training_batches),
            epochs=5,
            callbacks=[checkpoint, history],
            validation_data=val_generator,
            validation_steps=len(validation_batches))

        loss_filepath = "VGG-16+LSTM.timesteps_" + str(timestep) + ".lr_{}.loss.log".format(learning_rate)
        history.log_training_loss(loss_filepath)

        epoch_filepath = "VGG-16+LSTM.timesteps_" + str(timestep) + ".lr_{}.acc.log".format(learning_rate)
        history.log_epoch(epoch_filepath)

        if K.backend() == 'tensorflow':
            K.clear_session()


def parse_args():
    parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
    parser.add_argument('--timestep', dest='timestep',
                        help='VGG-16+LSTM model timestep',
                        default=10, type=int)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    if K.backend() == 'tensorflow':
      KTF.set_session(get_session(0.25))

    train_net(args.timestep)
