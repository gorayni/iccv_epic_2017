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


def load_images_batch(image_data_generator, users, batch, num_classes=21, target_size=(224, 224), batch_size=None):
    image_shape = target_size + (3,)

    if not batch_size:
        batch_size = batch.size
    batch_x = np.zeros((1, batch_size,) + image_shape, dtype=K.floatx())
    batch_y = np.zeros((1, batch_size, num_classes), dtype='float32')

    for i, ind in enumerate(batch.indices):
        image = users[batch.user_id][batch.date].images[ind]
        img = load_img(image.path, target_size=target_size, grayscale=False)
        x = img_to_array(img)
        x = image_data_generator.random_transform(x)
        x = image_data_generator.standardize(x)
        batch_x[0, i] = x
        batch_y[0, i, image.label] = 1.
    return batch_x, batch_y


def generate_batch(image_data_generator, users, batches, steps_per_epoch=None, num_classes=21, target_size=(224, 224)):
    if not steps_per_epoch:
        steps_per_epoch = len(batches)
    while True:
        np.random.shuffle(batches)
        for i in range(steps_per_epoch):
            batch_x, batch_y = load_images_batch(image_data_generator, users, batches[i], num_classes, target_size)
            yield (batch_x, batch_y)
