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
        self.note_classes = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B", "REST"]
        self.duration_classes = ["1", "1/2", "1/4", "1/8", "1/16", "1/32", "1/64", "1/128"]
        self.remove_empty_classes()

    def remove_empty_classes(self):
        examples_per_class=np.sum(self.train.labels.notes, axis=0)
        empty_classes=np.argwhere(examples_per_class==0).flatten()
        for cls in empty_classes:
            self.train.labels.notes=np.delete(self.train.labels.notes, cls, 1)
            self.test.labels.notes=np.delete(self.test.labels.notes, cls, 1)
            del self.note_classes[cls]
        examples_per_class=np.sum(self.train.labels.duration, axis=0)
        empty_classes=np.argwhere(examples_per_class == 0).flatten()
        for cls in empty_classes:
            self.train.labels.duration=np.delete(self.train.labels.duration, cls, 1)
            self.test.labels.duration=np.delete(self.test.labels.duration, cls, 1)
            del self.duration_classes[cls]

    def get_weights(self):
        def get_weight(data):
            freqs = np.mean(data,axis=0)+1e-8
            i_freqs = 1. / freqs
            weight_vector = freqs.shape[0] * i_freqs / np.sum(i_freqs)
            return weight_vector

        weight_vectors = {
            "train_notes": get_weight(self.train.labels.notes),
            "train_duration": get_weight(self.train.labels.duration),
            "test_notes": get_weight(self.test.labels.notes),
            "test_duration": get_weight(self.test.labels.duration)
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
