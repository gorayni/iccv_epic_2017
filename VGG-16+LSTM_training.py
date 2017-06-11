from __future__ import division

import numpy as np
import ntcir
import ntcir.IO as IO

from keras import backend as K
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
from experiments.utils import HistoryLog
from experiments.utils import generate_batch
import experiments as exp

users = IO.load_annotations(ntcir.filepaths)
categories = IO.load_categories(ntcir.filepaths)
sorted_users = ntcir.utils.sort(users)

# Full day sequences
num_frames_per_day = 2880
sequences = ntcir.get_sequences(sorted_users, num_frames_per_day)


training_set = ntcir.read_split('training_split.txt')
validation_set = ntcir.read_split('validation_split.txt')

training_batches = ntcir.get_training_batches(training_set, sequences)
validation_batches = ntcir.get_batches(validation_set, sequences)

for learning_rate in [0.0001, 0.00005, 0.000025, 0.000075]:
  K.set_learning_phase(1)

  np.random.seed(42)
  learning_rate=0.00001
  sgd = SGD(lr=learning_rate, decay=0.000005, momentum=0.9, nesterov=True)
  model = exp.vgg_16_plus_lstm(vgg16_weights='weights.VGG-16.best.hdf5')
  model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

  # prepare data augmentation configuration
  train_datagen = ImageDataGenerator(rescale=1./255,
                                     rotation_range=40,
                                     width_shift_range=0.2,
                                     height_shift_range=0.2,
                                     zoom_range=0.2,
                                     horizontal_flip=True)

  val_datagen = ImageDataGenerator(rescale=1./255)

  train_generator = generate_batch(train_datagen, users, training_batches)

  val_generator = generate_batch(val_datagen, users, validation_batches)
      
  # checkpoint
  weights_filepath="weights.VGG-16+LSTM.lr_"+str(learning_rate)+".{epoch:02d}.hdf5"
  checkpoint = ModelCheckpoint(weights_filepath, monitor='val_acc', verbose=1)
  history = HistoryLog()

  # fine-tune the model
  model.fit_generator(
          train_generator,
          steps_per_epoch=len(training_batches),#36095,#15,
          epochs=5,
          callbacks=[checkpoint, history],
          validation_data=val_generator,
          validation_steps=len(validation_batches))#6225)#20)

  loss_filepath = "VGG-16+LSTM.lr_{}.loss.log".format(learning_rate)
  history.log_training_loss(loss_filepath)

  epoch_filepath = "VGG-16+LSTM.lr_{}.acc.log".format(learning_rate)
  history.log_epoch(epoch_filepath)

  K.clear_session()
