from __future__ import division
import re
import os
import numpy as np
from . import Image
from . import Date
from . import User

from collections import defaultdict


class Filepaths(object):
    def __init__(self, ntcir_dir):
        ntcir_dir = os.path.realpath(ntcir_dir)
        self.annotations = os.path.join(ntcir_dir, 'annotations.txt')
        self.labels = os.path.join(ntcir_dir, 'labels.txt')
        self.images_dir = os.path.join(ntcir_dir, 'images')


def load_labels(filepaths):
    return list(np.loadtxt(filepaths.labels, str, delimiter='\n'))


def load_annotations(filepaths):
    users = defaultdict(lambda: defaultdict(list))
    lines = np.loadtxt(filepaths.annotations, str, delimiter='\n')
    for i, line in enumerate(lines):
        path, label = line.rsplit(' ', 1)
        user, date, time = path.split(os.path.sep)
        label = int(label)
        path = os.path.join(filepaths.images_dir, path)

        time = re.sub("[^0-9]", "", time.split('_')[-1])

        image = Image(path, time, label)
        users[user][date].append(image)

    sorted_users = list()
    for user, dates in users.iteritems():
        sorted_dates = list()
        for date, images in dates.iteritems():
            images.sort(key=lambda img: img.time)
            sorted_dates.append(Date(date, images))
        sorted_dates.sort(key=lambda date: date.date)

        sorted_users.append(User(user, sorted_dates))
    sorted_users.sort(key=lambda user: user.user)
    return sorted_users
