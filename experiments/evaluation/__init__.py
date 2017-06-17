from __future__ import division
import sklearn.metrics as metrics
import numpy as np
from easydict import EasyDict as edict


def predict(model, features, users, sequences, split_set, window_size=10):
    predictions = list()
    groundtruth = list()
    for user_id, date in split_set:
        for seq in sequences[user_id][date]:
            for start_ind in range(seq.start, seq.end, window_size):

                if start_ind + window_size > seq.end:
                    end_ind = seq.end
                else:
                    end_ind = start_ind + window_size
                model.reset_states()
                for ind in range(start_ind, end_ind):
                    target = users[user_id][date].images[ind].label

                    x = features[user_id][date][ind][np.newaxis, np.newaxis, ...]
                    probabilities = model.predict(x)
                    predictions.append(np.argmax(probabilities))
                    groundtruth.append(target)
    return predictions, groundtruth


def test(model, features, users, sequences, split_set, window_size=10):
    predictions, groundtruth = predict(model, features, users, sequences, split_set, window_size)
    accuracy = metrics.accuracy_score(groundtruth, predictions)
    return predictions, groundtruth, accuracy


def evaluate(groundtruth, predictions):
    return edict({'accuracy': metrics.accuracy_score(groundtruth, predictions),
                  'macro_precision': metrics.precision_score(groundtruth, predictions, average='macro'),
                  'macro_recall': metrics.recall_score(groundtruth, predictions, average='macro'),
                  'macro_f1': metrics.f1_score(groundtruth, predictions, average='macro'),
                  'recall': metrics.recall_score(groundtruth, predictions, average=None)
                  })
