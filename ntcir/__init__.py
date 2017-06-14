from __future__ import division

import os
import numpy as np
import utils
from collections import namedtuple
from collections import defaultdict

Sequence = namedtuple('Sequence', 'start end')


class Image(object):
    def __init__(self, path, time, label):
        self.path = os.path.realpath(path)
        self.time = time
        self.label = label

    def __repr__(self):
        return 'Image(path: ' + repr(self.path) + ', time: ' + repr(self.time) + ', label: ' + repr(self.label) + ')'


class Day(object):
    def __init__(self, date, images, user=None):
        self.date = date
        self.images = images
        self.user = user

    @property
    def num_images(self):
        return len(self.images)

    def __repr__(self):
        return 'Date(date: ' + repr(self.date) + ', Images: ' + repr(self.images) + ')'

    def __eq__(self, other):
        return self.date == other.date

    def __cmp__(self, other):
        return cmp(self.date, other.date)

    def __len__(self):
        return self.num_images

    def __add__(self, other):
        return self.num_images + other

    def __radd__(self, other):
        return other + self.num_images


class User(object):
    def __init__(self, id_, days):
        self.id_ = id_
        self.days = days

        for day in days:
            day.user = self

    @property
    def num_images(self):
        return sum([d.num_images for d in self.days])

    def __repr__(self):
        return 'User(id: ' + repr(self.id_) + ', Days: ' + repr(self.days) + ')'

    def __eq__(self, other):
        return self.id_ == other.id_


class Batch(object):
    def __init__(self, user_id, date, indices):
        self.user_id = user_id
        self.date = date
        self.indices = indices

    @property
    def size(self):
        return len(self.indices)

    def __repr__(self):
        return 'Batch(user_id: ' + repr(self.user_id) + ', date: ' + repr(self.date) + ', indices: ' + repr(
            self.indices) + ')'


def get_sequences(users, max_minute_separation=5):
    sequences = defaultdict(lambda: defaultdict(list))
    for user in users:
        for i, day in enumerate(user.days):
            times = np.asarray([utils.time2sec(img.time) for img in day.images])
            time_diff = np.diff(times) / 60

            start_ind = 0
            for ind in np.where(time_diff > max_minute_separation)[0]:
                seq = Sequence(start_ind, ind + 1)
                sequences[user.id_][day.date].append(seq)
                start_ind = ind + 1

            seq = Sequence(start_ind, day.num_images)
            sequences[user.id_][day.date].append(seq)
    return utils.default_to_regular(sequences)


def read_split(filepath):
    days = list()
    with open(filepath) as f:
        for line in f.readlines():
            user_id, date = line.replace("\n", "").split(' ')
            days.append((user_id, date))
    return days


def get_training_batches(training_set, sequences, batch_size=10):
    batches = list()
    for user_id, date in training_set:
        for seq in sequences[user_id][date]:
            if seq.end - seq.start > batch_size:
                num_windows = seq.end - seq.start - batch_size + 1
                window_size = batch_size
            else:
                num_windows = 1
                window_size = seq.end - seq.start

            for start_ind in range(num_windows):
                indices = np.arange(start_ind, start_ind + window_size) + seq.start
                b = Batch(user_id, date, indices)
                batches.append(b)
    return batches


def get_batches(split_set, sequences, batch_size=10, overlap=0):
    batches = list()
    for user_id, date in split_set:
        for s in sequences[user_id][date]:
            start_ind = s.start
            end_ind = start_ind + batch_size
            while end_ind <= s.end:
                indices = np.arange(start_ind, end_ind)
                b = Batch(user_id, date, indices)
                batches.append(b)

                start_ind = end_ind - overlap
                end_ind = start_ind + batch_size
    return batches


def get_piggyback_batches(split_set, sequences, batch_size=10, overlap=2):
    sequence_batches = list()
    non_overlaping = batch_size - overlap
    for user_id, date in split_set:
        for s in sequences[user_id][date]:
            for start_ind in range(s.start, s.end - batch_size):
                batches = list()
                end_ind = (int((s.end - start_ind - batch_size) / non_overlaping) + 1) * non_overlaping + start_ind
                for ind in range(start_ind, end_ind, non_overlaping):
                    indices = np.arange(ind, ind + batch_size)
                    b = Batch(user_id, date, indices)
                    batches.append(b)
                sequence_batches.append(batches)
    return sequence_batches


import IO

filepaths = IO.Filepaths(ntcir_dir='datasets/ntcir')
