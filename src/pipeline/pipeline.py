#!/usr/bin/python3
import json
import os
from datetime import datetime
import shutil
import tensorflow as tf
from models.my_model import MyModel

def expand_config(config):
    config["outputs"] = {
            "semitone": {
                "loss": "wcce",
                "metrics": ["ba", "ac"],
                "loss_weight": 154/1623

            },
            "octave": {
                "loss": "wcce",
                "metrics": ["ba", "ac"],
                "loss_weight": 182/1623

            },
            "dur_log": {
                "loss": "wcce",
                "metrics": ["ba", "ac"],
                "loss_weight": 286/1623

            },
            "dotted": {
                "loss": "wcce",
                "metrics": ["ba", "ac"],
                "loss_weight": 1001/1623

            }
        }
    return config


class Pipeline:
    def __init__(
            self,
            config_file_path: str,
            output_path_parent: str,
            verbose: int
    ):
        config_file_name = os.path.basename(config_file_path).split(".")[0]
        with open(config_file_path) as fp:
            self.run_configs: list = json.load(fp)

        current_date = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        folder_name = f'{config_file_name}({current_date})'
        self.output_path = os.path.join(
            output_path_parent,
            folder_name
        )
        os.makedirs(self.output_path, exist_ok=True)
        shutil.copy(config_file_path, self.output_path)
        self.verbose = verbose

    def run(self):
        reports = []
        gpus = tf.config.experimental.list_physical_devices('GPU')
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        for run_config in self.run_configs:
            reports.append(MyModel(run_config, self.output_path, self.verbose).run())
        return reports
