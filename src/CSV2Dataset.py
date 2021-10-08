import json
import math
import os
from configparser import ConfigParser

import numpy as np
from utils import get_file_paths


class CSV2Dataset:
    def __init__(self, create=True, save=False):
        config = ConfigParser()
        config.read('.\config.ini')

        config_group = "CSV2Dataset"

        input_path = config.get(config_group, 'input_path')
        dummy_path = config.get(config_group, 'dummy_path')
        dummy = int(config.get(config_group, 'dummy'))
        self.input_path = dummy_path if dummy else input_path
        self.output_path = config.get(config_group, 'output_path')
        self.input_beats = int(config.get(config_group, 'input_beats'))
        self.output_beats = int(config.get(config_group, 'label_beats'))
        self.window_beats = self.input_beats + self.output_beats
        if create:
            self.create_dataset("test")
            self.create_dataset("train")
            if save:
                self.save_dataset()
        else:
            self.read_dataset()

    def read_dataset(self):
        self.train_inputs = np.loadtxt(os.path.join(self.output_path, "train_inputs.csv")).reshape(-1, self.input_beats,
                                                                                                   13)
        self.test_inputs = np.loadtxt(os.path.join(self.output_path, "test_inputs.csv")).reshape(-1, self.input_beats,
                                                                                                 13)
        self.train_labels = {
            "notes": np.loadtxt(os.path.join(self.output_path, "train_label_notes.csv"))
            , "duration": np.loadtxt(os.path.join(self.output_path, "train_label_duration.csv"))
            # , "repeated_note": np.loadtxt(os.path.join(self.output_path, "train_label_repeated_note.csv"))
            # , "repeated_duration": np.loadtxt(os.path.join(self.output_path, "train_label_repeated_duration.csv"))
        }
        self.test_labels = {
            "notes": np.loadtxt(os.path.join(self.output_path, "test_label_notes.csv")),
            "duration": np.loadtxt(os.path.join(self.output_path, "test_label_duration.csv"))
            # , "repeated_note": np.loadtxt(os.path.join(self.output_path, "test_label_repeated_note.csv"))
            # , "repeated_duration": np.loadtxt(os.path.join(self.output_path, "test_label_repeated_duration.csv"))
        }
        print("")

    def save_dataset(self):
        if not os.path.exists(self.output_path):
            os.mkdir(self.output_path)
        array_files = [
            (self.train_inputs.reshape(-1, 13), "train_inputs", '%1.6f'),
            (self.train_labels["notes"], "train_label_notes", '%i'),
            (self.train_labels["duration"], "train_label_duration", '%i'),
            (self.train_labels["repeated_note"], "train_label_repeated_note", '%i'),
            (self.train_labels["repeated_duration"], "train_label_repeated_duration", '%i'),
            (self.test_inputs.reshape(-1, 13), "test_inputs", '%1.6f'),
            (self.test_labels["notes"], "test_label_notes", '%i'),
            (self.test_labels["duration"], "test_label_duration", '%i'),
            # (self.test_labels["repeated_note"], "test_label_repeated_note", '%i'),
            # (self.test_labels["repeated_duration"], "test_label_repeated_duration", '%i')
        ]
        for data, name, style in array_files:
            np.savetxt(os.path.join(self.output_path, f'{name}.csv'), data, fmt=style)

    def create_dataset(self, target):
        file_names = get_file_paths(os.path.join(self.input_path, target))
        inputs = []
        labels = []
        for file_name in file_names:
            contents = np.loadtxt(file_name)
            # add inputs and labels by sliding window
            for i in range(contents.shape[0] - self.window_beats):
                repeated_note = np.argmax(contents[i + self.input_beats - 1]) == np.argmax(
                    contents[i + self.input_beats])
                repeated_duration = np.max(contents[i + self.input_beats - 1]) == np.max(contents[i + self.input_beats])
                inputs.append(contents[i:i + self.input_beats])
                # labels.append(Label(contents[i + self.input_beats], repeated_note, repeated_duration))
                labels.append(Label(contents[i + self.input_beats]))

        inputs = np.array(inputs)
        labels = \
            {
                "notes": np.array([label.notes for label in labels]),
                "duration": np.array([label.duration for label in labels]),
                # "repeated_note": np.array([label.repeated_note for label in labels]),
                # "repeated_duration": np.array([label.repeated_duration for label in labels])
            }
        if target == 'train':
            self.train_inputs = inputs
            self.train_labels = labels
        elif target == 'test':
            self.test_inputs = inputs
            self.test_labels = labels


class Label:
    #def __init__(self, beat, repeated_note, repeated_duration):
    def __init__(self, beat):
        self.notes = np.sign(beat)
        duration = np.zeros(8)
        duration[-int(round(math.log2(np.max(beat))))] = 1
        self.duration = duration
        # self.repeated_note = repeated_note
        # self.repeated_duration = repeated_duration


if __name__ == "__main__":
    CSV2Dataset(create=False)
