#!/usr/bin/python3
import csv
import json
import os
from datetime import datetime
import shutil
from tensorflow.python import keras
import tensorflow as tf
from architecture import losses, metrics, layers, optimizers, preprocessing
from data_management import dataset


def compute_metrics(model, inp, output, batch_size, run_name):

    evaluation = model.evaluate(
        inp,
        output,
        batch_size
    )
    row = [run_name] + evaluation
    headers = ["run_name"] + model.metrics_names
    return row, headers


class RunConfig:
    def __init__(self, config):
        self.run_name = config["run_name"]
        self.optimizer = optimizers.get_optimizer(config["optimizer"])
        self.batch_size = config["batch_size"]
        self.max_epochs = config["max_epochs"]
        self.input_beats = config["input_beats"]
        raw_loss_weights = config["loss_weights"]
        active_features = {
            feature: raw_loss_weights[feature] > 0 for feature in raw_loss_weights.keys()
        }
        self.data = dataset.Dataset(self.input_beats, self.batch_size, active_features)
        self.model: keras.Model = layers.get_model(
            config["model"],
            self.data.input_shape,
            self.data.number_of_classes
        )

        metric_name = config["metric_name"]
        loss_name = config["loss_function_name"]
        self.metrics = {}
        self.losses = {}
        self.loss_weights = {}
        for feature in self.data.number_of_classes.keys():
                self.metrics[feature] = \
                    [
                        metrics.get_metric(name, self.data.number_of_classes[feature]) for name in metric_name
                    ]
                self.losses[feature] = losses.get_loss_function(
                    loss_name
                )

                self.loss_weights[feature] = 1./len(self.data.number_of_classes.keys())
        print()
class ModelTrainer:
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

    def save_weights(self, model, name):
        folder_path = os.path.join(
            self.output_path,
            "weights"
        )
        os.makedirs(folder_path, exist_ok=True)
        weight_filename = os.path.join(
            folder_path,
            f'{name}.h5'
        )
        model.save_weights(weight_filename)

    def run_one(self, config):
        rc = RunConfig(
            config,
        )
        rc.model.compile(
            optimizer=rc.optimizer,
            loss_weights=rc.loss_weights,
            loss=rc.losses,
            metrics=rc.metrics
        )
        log_folder = os.path.join(
            self.output_path,
            "logs"
        )
        tensorboard = keras.callbacks.TensorBoard(
            log_dir=os.path.join(log_folder, rc.run_name)
        )

        rc.model.fit(
            rc.data.train,
            epochs=rc.max_epochs,
            verbose=self.verbose,
            validation_data=rc.data.test,
            callbacks=[tensorboard]
        )
        self.save_weights(rc.model, rc.run_name)
        return compute_metrics(rc.model, rc.test_input, rc.test_output, rc.batch_size, rc.run_name)

    def run_all(self):
        rows = []
        for config in self.run_configs:
            row, header = self.run_one(config)
            rows.append(header)
            rows.append(row)
        with open(
                os.path.join(
                    self.output_path,
                    "metrics_report.csv"
                ),
                'w',
        ) as csv_file:
            writer = csv.writer(csv_file)
            writer.writerows(rows)
