import json
import os
import numpy as np

from src.data_processing import dataset, utils


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

        self.input_path = os.path.join(*(params["input_path"].split("\\")[0].split("/")))
        self.output_path = os.path.join(*(params["output_path"].split("\\")[0].split("/")))
        self.datasets = {}

    def get_dataset(self, input_beats, output_beats):
        dataset_name = f'i_{input_beats}_l{output_beats}'
        data = self.datasets.get(dataset_name)
        if data is None:
            return self.extract_dataset(input_beats, output_beats)
        else:
            return data

    def extract_distribution(self, distribution_name, input_beats, window_beats):
        file_names = utils.get_file_paths(os.path.join(self.input_path, distribution_name))
        inputs = []
        label_notes = []
        label_duration = []
        for file_name in file_names:
            contents = np.loadtxt(file_name)
            # add inputs and labels by sliding window
            for i in range(contents.shape[0] - window_beats + 1):
                inputs.append(contents[i:i + input_beats])
                label_beat = contents[i + input_beats]

                label_notes.append(label_beat[:13])
                label_duration.append(label_beat[13:])

        inputs = np.array(inputs)
        labels = dataset.Labels(np.array(label_notes), np.array(label_duration))
        return dataset.Distribution(inputs, labels)

    def extract_dataset(self, input_beats, output_beats):
        window_beats = input_beats + output_beats
        return dataset.Dataset(
            self.extract_distribution("train", input_beats, window_beats),
            self.extract_distribution("test", input_beats, window_beats),
        )

    def get_average_lengths(self):
        file_names = utils.get_file_paths(os.path.join(self.input_path, "train"))
        print(np.mean([np.loadtxt(file_name).shape[0] for file_name in file_names]))
