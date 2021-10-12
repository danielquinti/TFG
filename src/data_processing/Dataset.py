import os
import numpy as np


class Labels:
    def __init__(self, notes, duration):
        self.notes = notes
        self.duration = duration


class Distribution:
    def __init__(self, inputs, labels: Labels):
        self.inputs = inputs
        self.labels = labels


def load_dataset(output_path,
                 input_beats):
    return Dataset(
        Distribution(
            np.loadtxt(os.path.join(output_path, "train_inputs.csv")).reshape((-1, input_beats, 13)),
            Labels(
                np.loadtxt(os.path.join(output_path, "train_label_notes.csv")),
                np.loadtxt(os.path.join(output_path, "train_label_duration.csv"))
            )
        ),
        Distribution(
            np.loadtxt(os.path.join(output_path, "test_inputs.csv")).reshape((-1, input_beats, 13)),
            Labels(
                np.loadtxt(os.path.join(output_path, "test_label_notes.csv")),
                np.loadtxt(os.path.join(output_path, "test_label_duration.csv"))
            )
        ),
    )


class Dataset:
    def __init__(self, train: Distribution, test: Distribution):
        self.train = train
        self.test = test

    def get_weights(self):
        def get_weight(data, n_classes):
            aux_data = np.argmax(data, axis=1).flatten()
            freqs = np.histogram(aux_data, bins=range(n_classes + 1), density=True)[0]
            freqs += 0.00001
            i_freqs = 1. / freqs
            weight_vector = n_classes * i_freqs / np.sum(i_freqs)
            return weight_vector

        weight_vectors = {
            "train_notes": get_weight(self.train.labels.notes, 13),
            "train_duration": get_weight(self.train.labels.duration, 8),
            "test_notes": get_weight(self.test.labels.notes, 13),
            "test_duration": get_weight(self.test.labels.duration, 8)
        }
        return weight_vectors

    def save(self, output_path):
        if not os.path.exists(output_path):
            os.mkdir(output_path)
        np.savetxt(
            os.path.join(output_path, 'train_inputs.csv'),
            self.train.inputs.reshape(-1, 13),
            fmt='%1.6f')
        np.savetxt(
            os.path.join(output_path, 'train_label_notes.csv'),
            self.train.labels.notes,
            fmt='%i')
        np.savetxt(
            os.path.join(output_path, 'train_label_duration.csv'),
            self.train.labels.duration,
            fmt='%i')
        np.savetxt(
            os.path.join(output_path, 'test_inputs.csv'),
            self.test.inputs.reshape(-1, 13),
            fmt='%1.6f')
        np.savetxt(
            os.path.join(output_path, 'test_label_notes.csv'),
            self.test.labels.notes,
            fmt='%i')
        np.savetxt(
            os.path.join(output_path, 'test_label_duration.csv'),
            self.test.labels.duration,
            fmt='%i')
