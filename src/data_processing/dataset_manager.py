import json
import os

import numpy as np

import dataset
from src import utils


class DatasetManager:
    def __init__(self):
        fp = open(
            os.path.join(
                "src",
                "config",
                "csv_to_dataset_config.json"
            )
        )
        params = json.load(fp)

        self.input_path = os.path.join(
            *(params["input_path"].split("/")),
            "npy"
        )
        self.output_path = os.path.join(*(params["output_path"].split("/")))
        self.raw_data = {
            "test": self.read_files("test"),
            "train": self.read_files("train")
        }
        self.datasets = {}

    def get_dataset(self, input_beats, output_beats, active_features: dict):

        key = f'i_{input_beats}_l{output_beats}_af_{json.dumps(active_features)}'
        data = self.datasets.get(key)
        if data is None:
            data = self.extract_dataset(input_beats, output_beats, active_features)
            self.datasets[key] = data
        return data

    def read_files(self, distribution_name):
        contents = []
        file_names = utils.get_file_paths(os.path.join(self.input_path, distribution_name))
        if not file_names:
            raise FileNotFoundError(f'Could not find files for {distribution_name} distribution in {self.input_path}.')
        for file_name in file_names:
            contents.append(np.load(file_name))
        return contents

    def extract_distribution(
            self,
            distribution_name,
            input_beats,
            window_beats,
            active_features: dict
    ):
        inputs = []
        labels = {feature: [] for feature, is_active in active_features.items() if is_active}
        for song in self.raw_data[distribution_name]:
            # add inputs and labels by sliding window
            for i in range(song.shape[0] - window_beats + 1):
                inputs.append(song[i:i + input_beats])
                label_beat = song[i + input_beats]
                if active_features["notes"]:
                    labels["notes"].append(label_beat[:13])
                if active_features["duration"]:
                    labels["duration"].append(label_beat[13:])

        inputs = np.array(inputs)
        labels = {feature: np.array(labels[feature]) for feature in labels.keys()}
        return dataset.Distribution(inputs, labels)

    def extract_dataset(self, input_beats, output_beats, active_features: dict):
        window_beats = input_beats + output_beats
        return dataset.Dataset(
            self.extract_distribution("train", input_beats, window_beats, active_features),
            self.extract_distribution("test", input_beats, window_beats, active_features),
        )
