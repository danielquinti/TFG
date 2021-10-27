import json
import math
from .utils import *
from .dataset import *


class DatasetManager:
    def __init__(self):
        fp = open(
            os.path.join(
                "src",
                "data_processing",
                "csv_to_dataset_config.json"
            )
        )
        params = json.load(fp)

        self.input_path = os.path.join(*(params["input_path"].split("\\")[0].split("/")))
        self.output_path = os.path.join(*(params["output_path"].split("\\")[0].split("/")))
        self.datasets = {}

    def get_dataset(self, input_beats, output_beats):
        dataset_name = f'i_{input_beats}_l{output_beats}'
        dataset = self.datasets.get(dataset_name)
        if dataset is None:
            return self.extract_dataset(input_beats, output_beats)
        else:
            return dataset

    def extract_distribution(self, distribution_name, input_beats, window_beats):
        file_names = get_file_paths(os.path.join(self.input_path, distribution_name))
        inputs = []
        label_notes = []
        label_duration = []
        for file_name in file_names:
            contents = np.loadtxt(file_name)
            # add inputs and labels by sliding window
            for i in range(contents.shape[0] - window_beats + 1):
                inputs.append(contents[i:i + input_beats])
                label_beat = contents[i + input_beats]

                duration = np.zeros(8)
                duration[-int(round(math.log2(np.max(label_beat))))] = 1

                label_notes.append(np.sign(label_beat))
                label_duration.append(duration)

        inputs = np.array(inputs)
        labels = Labels(np.array(label_notes), np.array(label_duration))
        return Distribution(inputs, labels)

    def extract_dataset(self, input_beats, output_beats):
        window_beats = input_beats + output_beats
        return Dataset(
            self.extract_distribution("train", input_beats, window_beats),
            self.extract_distribution("test", input_beats, window_beats),
        )

    # def load_dataset(self, input_beats, output_beats):
    #     dataset_name = f'i_{input_beats}_l{output_beats}'
    #     self.datasets[dataset_name] = Dataset(
    #         Distribution(
    #             np.loadtxt(os.path.join(self.output_path, "train_inputs.csv")).reshape((-1, input_beats, 13)),
    #             Labels(
    #                 np.loadtxt(os.path.join(self.output_path, "train_label_notes.csv")),
    #                 np.loadtxt(os.path.join(self.output_path, "train_label_duration.csv"))
    #             )
    #         ),
    #         Distribution(
    #             np.loadtxt(os.path.join(self.output_path, "test_inputs.csv")).reshape((-1, input_beats, 13)),
    #             Labels(
    #                 np.loadtxt(os.path.join(self.output_path, "test_label_notes.csv")),
    #                 np.loadtxt(os.path.join(self.output_path, "test_label_duration.csv"))
    #             )
    #         ),
    #     )
    #     return self.dataset
