import json
import math
import os

import numpy as np
from matplotlib import pyplot as plt

from utils import get_file_paths


class CSV2Dataset:
    def create_dataset(self,
                        input_path,
                        output_path,
                        input_beats,
                        label_beats,
                        save=False):
        window_beats = input_beats + label_beats
        self.__create__(input_path,
                            input_beats,
                            window_beats,
                            "test")
        self.__create__(input_path,
                            input_beats,
                            window_beats,
                            "train")
        self.get_weights()
        if save:
            self.__save__(output_path)
    def read_dataset(self,
                     output_path,
                     input_beats):
        self.train_inputs = np.loadtxt(os.path.join(output_path, "train_inputs.csv")).reshape(-1, input_beats,
                                                                                              13)
        self.test_inputs = np.loadtxt(os.path.join(output_path, "test_inputs.csv")).reshape(-1, input_beats,
                                                                                            13)
        self.train_labels = {
            "notes": np.loadtxt(os.path.join(output_path, "train_label_notes.csv")),
            "duration": np.loadtxt(os.path.join(output_path, "train_label_duration.csv"))
            # , "repeated_note": np.loadtxt(os.path.join(self.output_path, "train_label_repeated_note.csv"))
            # , "repeated_duration": np.loadtxt(os.path.join(self.output_path, "train_label_repeated_duration.csv"))
        }
        self.test_labels = {
            "notes": np.loadtxt(os.path.join(output_path, "test_label_notes.csv")),
            "duration": np.loadtxt(os.path.join(output_path, "test_label_duration.csv"))
            # , "repeated_note": np.loadtxt(os.path.join(self.output_path, "test_label_repeated_note.csv"))
            # , "repeated_duration": np.loadtxt(os.path.join(self.output_path, "test_label_repeated_duration.csv"))
        }
        self.get_weights()

    def __save__(self, output_path):
        if not os.path.exists(output_path):
            os.mkdir(output_path)
        array_files = [
            (self.train_inputs.reshape(-1, 13), "train_inputs", '%1.6f'),
            (self.train_labels["notes"], "train_label_notes", '%i'),
            (self.train_labels["duration"], "train_label_duration", '%i'),
            # (self.train_labels["repeated_note"], "train_label_repeated_note", '%i'),
            # (self.train_labels["repeated_duration"], "train_label_repeated_duration", '%i'),
            (self.test_inputs.reshape(-1, 13), "test_inputs", '%1.6f'),
            (self.test_labels["notes"], "test_label_notes", '%i'),
            (self.test_labels["duration"], "test_label_duration", '%i'),
            # (self.test_labels["repeated_note"], "test_label_repeated_note", '%i'),
            # (self.test_labels["repeated_duration"], "test_label_repeated_duration", '%i')
        ]
        for data, name, style in array_files:
            np.savetxt(os.path.join(output_path, f'{name}.csv'), data, fmt=style)

    def __create__(self,
                       input_path,
                       input_beats,
                       window_beats,
                       distribution):
        file_names = get_file_paths(os.path.join(input_path, distribution))
        inputs = []
        labels = []
        for file_name in file_names:
            contents = np.loadtxt(file_name)
            # add inputs and labels by sliding window
            for i in range(contents.shape[0] - window_beats):
                # repeated_note = np.argmax(contents[i + input_beats - 1]) == np.argmax(
                #     contents[i + input_beats])
                # repeated_duration = np.max(contents[i + input_beats - 1]) == np.max(contents[i + input_beats])
                # labels.append(Label(contents[i + input_beats], repeated_note, repeated_duration))

                inputs.append(contents[i:i + input_beats])
                labels.append(Label(contents[i + input_beats]))

        inputs = np.array(inputs)
        labels = \
            {
                "notes": np.array([label.notes for label in labels]),
                "duration": np.array([label.duration for label in labels]),
                # "repeated_note": np.array([label.repeated_note for label in labels]),
                # "repeated_duration": np.array([label.repeated_duration for label in labels])
            }
        if distribution == 'train':
            self.train_inputs = inputs
            self.train_labels = labels
        elif distribution == 'test':
            self.test_inputs = inputs
            self.test_labels = labels

    def get_weights(self):
        def get_weight(data,n_classes):
            aux_data = np.argmax(data, axis=1).flatten()
            freqs = np.histogram(aux_data, bins=range(n_classes+1),density=True)[0]
            freqs+=0.00001
            i_freqs=1./freqs
            weights = n_classes * i_freqs / np.sum(i_freqs)
            return weights
        self.weights={
            "train_notes": get_weight(self.train_labels["notes"],13),
            "train_duration" : get_weight(self.train_labels["duration"],8),
            "test_notes" : get_weight(self.test_labels["notes"],13),
            "test_duration" : get_weight(self.test_labels["duration"],8)
        }
        return self.weights
class Label:
    # def __init__(self, beat, repeated_note, repeated_duration):
    def __init__(self, beat):
        self.notes = np.sign(beat)
        duration = np.zeros(8)
        duration[-int(round(math.log2(np.max(beat))))] = 1
        self.duration = duration
        # self.repeated_note = repeated_note
        # self.repeated_duration = repeated_duration


if __name__ == "__main__":
    with open("config.json", "r") as fp:
        params = json.load(fp)
    CSV2Dataset(
        params['csv_to_dataset_input_paths'][params['dummy']],
        params['dummy'],
        params['csv_to_dataset_output_path'],
        params['input_beats'],
        params['label_beats'],
        params['create_dataset'],
        params['save_dataset'])
