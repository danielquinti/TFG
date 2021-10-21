import json
import math
from .utils import *
from .dataset import *


class DatasetManager:
    def __init__(self):
        self.dataset: Dataset = None
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
        self.input_beats = params["input_beats"]
        self.label_beats = params["label_beats"]
        self.window_beats = self.input_beats + self.label_beats

    def extract_dataset(self):
        distributions = {
            "train": None,
            "test": None
        }
        for distribution_name in distributions.keys():
            file_names = get_file_paths(os.path.join(self.input_path, distribution_name))
            inputs = []
            label_notes = []
            label_duration = []
            for file_name in file_names:
                contents = np.loadtxt(file_name)
                # add inputs and labels by sliding window
                for i in range(contents.shape[0] - self.window_beats+1):
                    inputs.append(contents[i:i + self.input_beats])
                    label_beat = contents[i + self.input_beats]

                    duration = np.zeros(8)
                    duration[-int(round(math.log2(np.max(label_beat))))] = 1

                    label_notes.append(np.sign(label_beat))
                    label_duration.append(duration)

            inputs = np.array(inputs)
            labels = Labels(np.array(label_notes), np.array(label_duration))
            distributions[distribution_name] = Distribution(inputs, labels)
        self.dataset = Dataset(distributions["train"], distributions["test"])
        return self.dataset

    def save_dataset(self):
        self.dataset.save(self.output_path)

    def load_dataset(self):
        self.dataset = Dataset(
            Distribution(
                np.loadtxt(os.path.join(self.output_path, "train_inputs.csv")).reshape((-1, self.input_beats, 13)),
                Labels(
                    np.loadtxt(os.path.join(self.output_path, "train_label_notes.csv")),
                    np.loadtxt(os.path.join(self.output_path, "train_label_duration.csv"))
                )
            ),
            Distribution(
                np.loadtxt(os.path.join(self.output_path, "test_inputs.csv")).reshape((-1, self.input_beats, 13)),
                Labels(
                    np.loadtxt(os.path.join(self.output_path, "test_label_notes.csv")),
                    np.loadtxt(os.path.join(self.output_path, "test_label_duration.csv"))
                )
            ),
        )
        return self.dataset
