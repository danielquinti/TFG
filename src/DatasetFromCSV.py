import numpy as np
from utils import get_file_names


class DatasetFromCSV:
    def __init__(self, input_width, label_width):
        self.input_width = input_width
        self.label_width = label_width
        self.window_width = input_width + label_width
        self.dataset = {
            "test": {
                "inputs": [],
                "labels": {
                    "notes": [],
                    "duration": []
                }

            },
            "train": {
                "inputs": [],
                "labels": {
                    "notes": [],
                    "duration": []
                }
            }
        }

        self.extract_data_from_csv("test")
        self.test_inputs = self.dataset["test"]["inputs"]
        self.test_labels = self.dataset["test"]["labels"]

        self.extract_data_from_csv("train")
        self.train_inputs = self.dataset["train"]["inputs"]
        self.train_labels = self.dataset["train"]["labels"]


    def extract_data_from_csv(self, target):
        file_names = get_file_names("data\\" + target)
        inputs = []
        labels = []
        for file_name in file_names:
            contents = np.loadtxt(file_name)
            # add inputs and labels by sliding window
            for i in range(contents.shape[0] - self.window_width):
                inputs.append(contents[i:i + self.input_width])
                labels.append(Label(contents[i + self.input_width]))
        inputs = np.array(inputs)
        labels = \
            {
                "notes": np.array([label.notes for label in labels]),
                "duration": np.array([label.duration for label in labels]),
            }
        self.dataset[target]["inputs"] = inputs
        self.dataset[target]["labels"] = labels


class Label:
    def __init__(self, beat):
        self.notes = np.sign(beat)
        self.duration = np.max(beat)
