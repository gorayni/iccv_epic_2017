from __future__ import division

import numpy as np
import ntcir
import ntcir.IO as IO

from keras import backend as K
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from experiments.utils import load_images_batch
import experiments as exp

users = IO.load_annotations(ntcir.filepaths)
categories = IO.load_categories(ntcir.filepaths)
sorted_users = ntcir.utils.sort(users)

# Full day sequences
num_frames_per_day = 2880
sequences = ntcir.get_sequences(sorted_users, num_frames_per_day)

training_set = ntcir.read_split('training_split.txt')
validation_set = ntcir.read_split('validation_split.txt')

overlap = 2
num_training_batches = 0
training_batches = list()
for user_id, date in training_set:
    batches = ntcir.get_batches([(user_id, date)], sequences, overlap=overlap)
    num_training_batches += len(batches)
    training_batches.append(batches)

num_validation_batches = 0
validation_batches = list()
for user_id, date in validation_set:
    batches = ntcir.get_batches([(user_id, date)], sequences, overlap=overlap)
    num_validation_batches += len(batches)
    validation_batches.append(batches)

weights_filepath = "weights.Filtered_VGG16+LSTM.lr_{lr:f}.{epoch:02d}.hdf5"
batch_size = timestep = 10
num_classes = 21

for learning_rate in [0.00001, 0.0001, 0.00005, 0.000025, 0.000075]:
    K.set_learning_phase(1)

    train_datagen = ImageDataGenerator(rescale=1. / 255,
                                       rotation_range=40,
                                       width_shift_range=0.2,
                                       height_shift_range=0.2,
                                       zoom_range=0.2,
                                       horizontal_flip=True)

    val_datagen = ImageDataGenerator(rescale=1. / 255)

    sgd = SGD(lr=learning_rate, decay=0.000005, momentum=0.9, nesterov=True)
    model = exp.filtered_vgg_16_plus_lstm('weights.VGG-16.best.hdf5')

    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    np.random.seed(42)

    train_acc = list()
    val_acc = list()
    loss = list()
    mask = np.ones((1, timestep, num_classes))
    prev_values = np.zeros((1, timestep, num_classes))
    for epoch in np.arange(5):

        epoch_train_acc = list()
        epoch_val_acc = list()

        np.random.shuffle(training_batches)
        for day_batches in training_batches:
            mask[:, -overlap:, :] = 1
            prev_values[:, -overlap:, :] = 0
            for batch in day_batches:
                batch_x, batch_y = load_images_batch(train_datagen, users, batch)
                batch_loss, batch_acc = model.train_on_batch([batch_x, mask, prev_values], [batch_y])
                prediction = model.predict_on_batch([batch_x, mask, prev_values])

                mask[:, -overlap:, :] = 0
                prev_values[:, -overlap:, :] = prediction[:, -overlap:, :]

                epoch_train_acc.append(batch_acc)
                loss.append(batch_loss)

        for day_batches in validation_batches:
            mask[:, -overlap:, :] = 1
            prev_values[:, -overlap:, :] = 0
            for batch in day_batches:
                batch_x, batch_y = load_images_batch(val_datagen, users, batch)
                batch_loss, batch_acc = model.test_on_batch([batch_x, mask, prev_values], [batch_y])
                prediction = model.predict_on_batch([batch_x, mask, prev_values])

                mask[:, -overlap:, :] = 0
                prev_values[:, -overlap:, :] = prediction[:, -overlap:, :]

                epoch_val_acc.append(batch_acc)

        epoch_train_acc = np.sum(epoch_train_acc) * batch_size / num_training_batches
        epoch_val_acc = np.sum(epoch_val_acc) * batch_size / num_validation_batches
        print 'Epoch: {}, Train Acc: {}, Validation Acc: {}'.format(epoch + 1, epoch_train_acc, epoch_val_acc)

        model.save(weights_filepath.format(lr=learning_rate, epoch=epoch + 1))

        train_acc.append(epoch_train_acc)
        val_acc.append(epoch_val_acc)

    loss = np.asarray(loss)
    train_acc = np.asarray(train_acc)
    val_acc = np.asarray(val_acc)

    np.savetxt('lstm.Filtered_VGG16+LSTM.lr_{}.acc.log'.format(learning_rate), np.vstack((train_acc, val_acc)).T,
               delimiter=",")
    np.savetxt('lstm.Filtered_VGG16+LSTM.lr_{}.loss.log'.format(learning_rate), loss.T, delimiter=",")
    K.clear_session()
