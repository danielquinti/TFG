import os

import numpy as np


class Distribution:
    def __init__(self, inputs, labels: dict):
        self.inputs = inputs
        self.labels = labels


def get_weights(data):
    freqs = np.mean(data, axis=0)
    i_freqs = np.divide(1., freqs, out=np.zeros_like(freqs, dtype='float'), where=freqs != 0)
    weight_vector = freqs.shape[0] * i_freqs / np.sum(i_freqs)
    return weight_vector


class Dataset:
    def __init__(self, train: Distribution, test: Distribution):
        self.train = train
        self.test = test
        self.features = train.labels.keys()
        self.remove_empty_classes()
        self.weights = {
            feature: get_weights(self.train.labels[feature]) for feature in self.features
        }
        self.number_of_classes = {
            feature: self.train.labels[feature].shape[-1] for feature in self.features
        }

    def remove_empty_classes(self):
        for feature in self.features:
            examples_per_class = np.sum(self.train.labels[feature], axis=0)
            empty_classes = np.argwhere(examples_per_class == 0).flatten()
            for cls in empty_classes:
                self.train.labels[feature] = np.delete(self.train.labels[feature], cls, 1)
                self.test.labels[feature] = np.delete(self.test.labels[feature], cls, 1)