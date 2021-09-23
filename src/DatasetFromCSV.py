import os
import math
import numpy as np
import random


def get_file_list(route):
    file_list = []
    for root, dirs, files in os.walk(route):
        for file in files:
            file_list.append(os.path.join(root, file))
    return file_list


class DatasetFromCSV:
    def __init__(self, input_width, label_width):
        self.input_width = input_width
        self.label_width = label_width
        self.window_width = input_width + label_width
        self.extract(0)
        self.extract(1)

    def extract(self, test):
        route = "data\\test" if test else "data\\train"
        file_names = get_file_list(route)
        inputs = []
        labels = []
        for file_name in file_names:
            contents = np.loadtxt(file_name)
            contents = contents[:(contents.shape[0] // self.window_width) * self.window_width]
            [inputs.append([contents[i + self.window_width * j] for i in range(self.input_width)]) for j in
             range(contents.shape[0] // self.window_width)]
            for j in range(1, math.ceil(contents.shape[0] / self.window_width) + 1):
                for i in range(self.label_width):
                    labels.append(Label(np.array(contents[i + self.window_width * j - self.label_width])))
        # Sample-level shuffle
        inputs = np.array(random.sample(inputs, len(inputs)))
        labels = random.sample(labels, len(labels))
        labels = \
            {
                "notes": np.array([[label.notes] for label in labels]),
                "duration": np.array([label.duration for label in labels]),
            }

        if test:
            self.test_inputs = inputs
            self.test_labels = labels
        else:
            self.train_inputs = inputs
            self.train_labels = labels


class Label:
    def __init__(self, beat):
        self.notes = beat
        self.duration = np.max(beat)
