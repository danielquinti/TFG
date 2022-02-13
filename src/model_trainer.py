#!/usr/bin/python3
import csv
import json
import os
from datetime import datetime
import shutil
from tensorflow.python import keras

from architecture import losses, metrics, layers, regularizers, optimizers
from data_processing import dataset_manager, dataset


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
    def __init__(self, config, data_manager):
        self.run_name = config["run_name"]
        self.model_name = config["model_name"]
        try:
            self.regularizer = regularizers.get_regularizer(config["regularizer"])
        except KeyError:
            self.regularizer = None
        self.optimizer = optimizers.get_optimizer(config["optimizer"])
        self.batch_size = config["batch_size"]
        self.max_epochs = config["max_epochs"]
        self.input_beats = config["input_beats"]
        self.label_beats = config["label_beats"]

        raw_loss_weights = config["loss_weights"]
        active_features = {
            feature: raw_loss_weights[feature] > 0 for feature in raw_loss_weights.keys()
        }
        self.data: dataset = data_manager.get_dataset(
            config["input_beats"],
            config["label_beats"],
            active_features
        )
        self.model: keras.Model = layers.get_model(
            self.model_name,
            self.data.train.inputs.shape[-1],
            self.data.number_of_classes,
            self.input_beats,
            self.label_beats,
            active_features,
            self.regularizer
        )
        self.train_input = self.data.train.inputs
        metric_names: dict = config["metric_names"]
        loss_names: dict = config["loss_function_names"]

        self.metrics = {}
        self.losses = {}
        self.train_output = {}
        self.test_output = []
        self.loss_weights = {}
        for feature, is_active in active_features.items():
            if is_active:
                self.metrics[feature] = \
                    [
                        metrics.get_metric(name, self.data.number_of_classes[feature]) for name in metric_names[feature]
                    ]
                self.losses[feature] = losses.get_loss_function(
                    loss_names[feature],
                    self.data.weights[feature]
                )
                self.test_output.append(self.data.test.labels[feature])
                self.train_output[feature] = self.data.train.labels[feature]
                self.loss_weights[feature] = raw_loss_weights[feature]
        self.test_input = self.data.test.inputs
        self.test_data = (
            self.data.test.inputs,
            self.test_output
        )


class ModelTrainer:
    def __init__(self, config_file_path: str):

        config_file_name = os.path.basename(config_file_path).split(".")[0]
        with open(config_file_path) as fp:
            params = json.load(fp)

        current_date = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        folder_name = f'{config_file_name}({current_date})'
        output_path_parent = os.path.join(
            *(params["output_path"].split("\\")[0].split("/"))
        )
        self.output_path = os.path.join(
            output_path_parent,
            folder_name
        )
        os.makedirs(self.output_path, exist_ok=True)
        shutil.copy(config_file_path, self.output_path)
        self.verbose = params["verbose"]
        self.model_configs: list = params["run_configs"]
        self.dataset_manager = dataset_manager.DatasetManager()

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
            self.dataset_manager
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
            x=rc.train_input,
            y=rc.train_output,
            epochs=rc.max_epochs,
            batch_size=rc.batch_size,
            verbose=self.verbose,
            shuffle=True,
            validation_data=rc.test_data,
            callbacks=[tensorboard]
        )
        self.save_weights(rc.model, rc.run_name)
        return compute_metrics(rc.model, rc.test_input, rc.test_output, rc.batch_size, rc.run_name)

    def run_all(self):
        rows = []
        for config in self.model_configs:
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
