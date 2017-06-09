from __future__ import division
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras import backend as K
import numpy as np

from keras.callbacks import Callback
from easydict import EasyDict as edict


class HistoryLog(Callback):
    def on_train_begin(self, logs={}):
        self.training = edict({'loss': []})
        self.epoch = edict({'acc': [], 'loss': [], 'val_acc': [], 'val_loss': []})

    def on_epoch_end(self, epoch, logs={}):
        self.epoch.acc.append(logs.get('acc'))
        self.epoch.loss.append(logs.get('loss'))
        self.epoch.val_acc.append(logs.get('val_acc'))
        self.epoch.val_loss.append(logs.get('val_loss'))

    def on_batch_end(self, batch, logs={}):
        self.training.loss.append(logs.get('loss'))

    def log_training_loss(self, fpath):
        training_loss = np.array(self.training.loss)
        np.savetxt(fpath, training_loss, delimiter=",")

    def log_epoch(self, fpath):
        epoch = np.asarray([self.epoch.loss, self.epoch.val_loss, self.epoch.acc, self.epoch.val_acc])
        np.savetxt(fpath, epoch.T, delimiter=",")


def load_images_batch(image_data_generator, users, batch, num_classes, target_size=(256, 256)):
    image_shape = target_size + (3,)

    batch_x = np.zeros((batch.size,) + image_shape, dtype=K.floatx())
    batch_y = np.zeros((batch.size, num_classes), dtype='float32')

    for i, ind in enumerate(batch.indices):
        image = users[batch.user_id][batch.date].images[ind]
        img = load_img(image.path, target_size=target_size, grayscale=False)
        x = img_to_array(img, dim_ordering='default')
        x = image_data_generator.random_transform(x)
        x = image_data_generator.standardize(x)
        batch_x[i] = x

        batch_y[i, image.label] = 1.
    return batch_x, batch_y
