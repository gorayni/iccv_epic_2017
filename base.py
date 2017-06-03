from __future__ import division

import collections
import multiprocessing
import os
import pickle

import caffe
import numpy as np
import sklearn.metrics as metrics
from sklearn.ensemble import RandomForestClassifier


def load_cnn(net_filepaths, input_size=1):
    # loads activities net
    net = caffe.Net(net_filepaths.deploy, net_filepaths.caffe_model, caffe.TEST)
    net.blobs['data'].reshape(input_size, 3, 227, 227)

    # Preprocessing for caffe inputs
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_transpose('data', (2, 0, 1))
    transformer.set_channel_swap('data', (2, 1, 0))
    transformer.set_raw_scale('data', 255.0)

    ilsvrc_mean = np.load(net_filepaths.image_mean)
    transformer.set_mean('data', ilsvrc_mean.mean(1).mean(1))  # mean pixel
    return net, transformer
