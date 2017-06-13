from __future__ import division

import numpy as np

from keras import backend as K
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
from experiments.utils import HistoryLog
import experiments as exp

img_width, img_height = 224, 224

seed = 42
np.random.seed(seed)

sgd = SGD(lr=0.000067, decay=0.005, momentum=0.9, nesterov=True)
model = exp.inceptionV3_first_phase_model()
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# prepare data augmentation configuration
train_datagen = ImageDataGenerator(rescale=1./255,
                                   rotation_range=40,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)

val_datagen = ImageDataGenerator(rescale=1./255)

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
weights_filepath="weights.InceptionV3.{epoch:02d}.hdf5"
checkpoint = ModelCheckpoint(weights_filepath, monitor='val_acc', verbose=1, save_best_only=False)
history = HistoryLog()

# fine-tune the model
model.fit_generator(
        train_generator,
        steps_per_epoch=36095,#36095,#15,
        epochs=10,
        callbacks=[checkpoint, history],
        validation_data=validation_generator,
        validation_steps=6225)#6225)#20)

loss_filepath = "InceptionV3.first_phase,loss.log"
history.log_training_loss(loss_filepath)

epoch_filepath = "InceptionV3.first_phase,epoch.log"
history.log_epoch(epoch_filepath)

K.clear_session()