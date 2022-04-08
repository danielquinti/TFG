#!/usr/bin/python3
import json
import os
from datetime import datetime
import shutil

from src.models.my_model import MyModel
import optuna


def expand_config(config):
    config["optimizer"] = {
        "name": "adam",
        "learning_rate": 1e-4
    }
    config["outputs"] = {
            "semitone": {
                "loss": "cce",
                "metrics": ["ba", "ac"],
                "loss_weight": 70 / 910

            },
            "octave": {
                "loss": "cce",
                "metrics": ["ba", "ac"],
                "loss_weight": 91 / 910

            },
            "dur_log": {
                "loss": "cce",
                "metrics": ["ba", "ac"],
                "loss_weight": 130 / 910

            },
            "dotted": {
                "loss": "cce",
                "metrics": ["ba", "ac"],
                "loss_weight": 455 / 910

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
            self.run_configs: list = [expand_config(x) for x in json.load(fp)]

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
        for run_config in self.run_configs:
            reports.append(MyModel(run_config, self.output_path, self.verbose).run())
        return reports