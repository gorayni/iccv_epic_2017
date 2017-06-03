from __future__ import division
import re
import os
import numpy as np
from . import Image
from . import Day
from . import User

from collections import defaultdict


class Filepaths(object):
    def __init__(self, ntcir_dir):
        ntcir_dir = os.path.realpath(ntcir_dir)
        self.annotations = os.path.join(ntcir_dir, 'annotations.txt')
        self.categories = os.path.join(ntcir_dir, 'categories.txt')
        self.images_dir = os.path.join(ntcir_dir, 'images')


def load_categories(filepaths):
    return list(np.loadtxt(filepaths.categories, str, delimiter='\n'))


def load_annotations(filepaths):
    users = defaultdict(lambda: defaultdict(list))
    lines = np.loadtxt(filepaths.annotations, str, delimiter='\n')
    for i, line in enumerate(lines):
        path, label = line.rsplit(' ', 1)
        id_, date, time = path.split(os.path.sep)
        label = int(label)
        path = os.path.join(filepaths.images_dir, path)

        time = re.sub("[^0-9]", "", time.split('_')[-1])

        image = Image(path, time, label)
        users[id_][date].append(image)

    sorted_users = list()
    for id_, dates in users.iteritems():
        sorted_days = list()
        for date, images in dates.iteritems():
            images.sort(key=lambda img: img.time)
            sorted_days.append(Day(date, images))
        sorted_days.sort(key=lambda day: day.date)

        sorted_users.append(User(id_, sorted_days))
    sorted_users.sort(key=lambda user: user.id_)
    return sorted_users
