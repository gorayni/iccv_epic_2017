from __future__ import division

import os
import numpy as np
import ntcir
import ntcir.IO as IO

import argparse

from keras import backend as K
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from experiments.utils import load_images_batch
import experiments as exp

def get_session(gpu_fraction=0.8):
    import tensorflow as tf

    num_threads = os.environ.get('OMP_NUM_THREADS')
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)

    if num_threads:
        return tf.Session(config=tf.ConfigProto(
            gpu_options=gpu_options, intra_op_parallelism_threads=num_threads))
    else:
        return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))


def train_net(timestep=10, overlap=2):
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

    model_fname = 'Filtered_VGG16+LSTM.timesteps_' + str(timestep) + '.overlap_' + str(overlap)
    weights_filepath = "weights." + model_fname + ".lr_{lr:f}.{epoch:02d}." + backend + ".hdf5"
    num_classes = 21

    num_training_batches = len(training_batches)
    num_validation_batches = len(validation_batches)

    mask = np.ones((1, timestep, 256))
    prev_values = np.zeros((1, timestep, 256))

    if timestep == 5:
        learning_rate = 0.000025
    else:
        learning_rate = 0.0001

    K.set_learning_phase(1)

    train_datagen = ImageDataGenerator(rescale=1. / 255,
                                       rotation_range=40,
                                       width_shift_range=0.2,
                                       height_shift_range=0.2,
                                       zoom_range=0.2,
                                       horizontal_flip=True)

    val_datagen = ImageDataGenerator(rescale=1. / 255)

    sgd = SGD(lr=learning_rate, decay=0.000005, momentum=0.9, nesterov=True)
    model = exp.filtered_vgg_16_plus_lstm_first_phase(base_model_weights, timestep=timestep)

    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    np.random.seed(42)

    train_acc = list()
    val_acc = list()
    loss = list()
    for epoch in np.arange(10):

        epoch_train_acc = list()
        epoch_val_acc = list()

        np.random.shuffle(training_batches)
        for batch in training_batches:
            batch_x, batch_y = load_images_batch(train_datagen, users, batch)
            batch_loss, batch_acc = model.train_on_batch([batch_x, mask, prev_values], [batch_y])

            epoch_train_acc.append(batch_acc)
            loss.append(batch_loss)

        for batch in validation_batches:
            batch_x, batch_y = load_images_batch(val_datagen, users, batch)
            batch_loss, batch_acc = model.test_on_batch([batch_x, mask, prev_values], [batch_y])

            epoch_val_acc.append(batch_acc)

        epoch_train_acc = np.sum(epoch_train_acc) / num_training_batches
        epoch_val_acc = np.sum(epoch_val_acc) / num_validation_batches
        print 'Epoch: {}, Train Acc: {}, Validation Acc: {}'.format(epoch + 1, epoch_train_acc, epoch_val_acc)

        model.save(weights_filepath.format(lr=learning_rate, epoch=epoch + 1))

        train_acc.append(epoch_train_acc)
        val_acc.append(epoch_val_acc)

    loss = np.asarray(loss)
    train_acc = np.asarray(train_acc)
    val_acc = np.asarray(val_acc)

    np.savetxt(model_fname + '.lr_{}.acc.log'.format(learning_rate), np.vstack((train_acc, val_acc)).T,
               delimiter=",")
    np.savetxt(model_fname + '.lr_{}.loss.log'.format(learning_rate), loss.T, delimiter=",")

    if K.backend() == 'tensorflow':
        K.clear_session()


def parse_args():
    parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
    parser.add_argument('--overlap', dest='overlap',
                        help='Filtered VGG-16+LSTM model overlap between batches',
                        default=2, type=int)
    parser.add_argument('--timestep', dest='timestep',
                        help='Filtered VGG-16+LSTM model timestep',
                        default=10, type=int)
    return parser.parse_args()


if __name__ == '__main__':

    if K.backend() == 'tensorflow':
        import keras.backend.tensorflow_backend as KTF
        KTF.set_session(get_session(0.33))

    args = parse_args()
    train_net(args.timestep, args.overlap)
