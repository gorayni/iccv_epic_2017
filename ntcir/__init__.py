from __future__ import division

import os
import numpy as np
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
        self.id_= id_
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

def time2ind(time):
    hour, minute, second =  [int(time[i:i+2]) for i in range(0, 6, 2)]
    index = 120*hour+2*minute+(1 if second > 30 else 0)
    return index

def time2sec(time):
    hour, minute, second =  [int(time[i:i+2]) for i in range(0, 6, 2)]
    index = 3600*hour+60*minute+second
    return index

def get_sequences(users, max_minute_separation=5):
    
    sequences = defaultdict(lambda: defaultdict(list))
    for user in users:    
        
        for i, day in enumerate(user.days):        
            times = np.asarray([time2sec(img.time) for img in day.images])
            time_diff = np.diff(times)/60
            
            start_ind = 0
            for ind in np.where(time_diff>max_minute_separation)[0]:
                seq = Sequence(start_ind, ind+1)
                sequences[user.id_][day.date].append(seq)
                start_ind = ind+1

            seq = Sequence(start_ind, day.num_images)
            sequences[user.id_][day.date].append(seq)            
    return sequences

import IO

filepaths = IO.Filepaths(ntcir_dir='datasets/ntcir')