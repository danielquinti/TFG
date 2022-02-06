#!/usr/bin/python3
import csv
import json
import os
from datetime import datetime

from tensorflow.python import keras

from architecture import losses, metrics, layers, regularizers, optimizers
from data_processing import dataset_manager, dataset


def compute_metrics(model, data, batch_size, run_name):
    evaluation = model.evaluate(
        data.test.inputs,
        [
            data.test.labels.notes,
            data.test.labels.duration
        ],
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
        self.data = data_manager.get_dataset(
                config["input_beats"],
                config["label_beats"]
            )
        raw_loss_weights = config["loss_weights"]
        notes_active = raw_loss_weights["notes"] > 0
        duration_active = raw_loss_weights["duration"] > 0
        self.model = layers.get_model(
            self.model_name,
            self.data.n_classes,
            self.data.d_classes,
            self.input_beats,
            self.label_beats,
            notes_active,
            duration_active,
            self.regularizer
        )
        self.train_input = self.data.train.inputs
        metric_names: dict = config["metric_names"]
        loss_names: dict = config["loss_function_names"]

        self.metrics = {}
        self.losses = {}
        self.train_output={}
        val_output=[]
        self.loss_weights={}
        if notes_active:
            self.metrics["notes"]=\
                [
                    metrics.get_metric(name, self.data.n_classes) for name in metric_names["notes"]
                ]
            self.losses["notes"] = losses.get_loss_function(
                loss_names["notes"],
                self.data.notes_weights
            )
            val_output.append(self.data.test.labels.notes)
            self.train_output['notes'] = self.data.train.labels.notes
            self.loss_weights["notes"] = raw_loss_weights["notes"]
        if duration_active:
            self.metrics["duration"] = \
                [
                    metrics.get_metric(name, self.data.d_classes) for name in metric_names["duration"]
                ]
            self.losses["duration"] = losses.get_loss_function(
                    loss_names["duration"],
                    self.data.duration_weights
                )
            val_output.append(self.data.test.labels.duration)
            self.train_output['duration'] = self.data.train.labels.duration
            self.loss_weights["duration"] = raw_loss_weights["duration"]
        self.validation_data = (
            self.data.test.inputs,
            val_output
        )


class ModelTrainer:
    def __init__(self):
        with open(
                os.path.join(
                    "src",
                    "config",
                    "train_config.json"
                )
        ) as fp:
            params = json.load(fp)

        current_date = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        folder_name = f'{current_date}({params["config_name"]})'
        output_path_parent = os.path.join(
            *(params["output_path"].split("\\")[0].split("/"))
        )
        if not os.path.exists(output_path_parent):
            os.mkdir(output_path_parent)
        self.output_path = os.path.join(
            output_path_parent,
            folder_name
        )
        self.verbose = params["verbose"]
        self.model_configs: list = params["run_configs"]
        self.dataset_manager = dataset_manager.DatasetManager()
        self.trained_models = {}

    def save_weights(self, model, name):
        folder_path = os.path.join(
            self.output_path,
            "weights"
        )
        if not os.path.exists(folder_path):
            os.mkdir(folder_path)
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
        tensorboard = keras.callbacks.TensorBoard(
            log_dir=os.path.join(self.output_path, rc.run_name))

        rc.model.fit(
            x=rc.train_input,
            y=rc.train_output,
            epochs=rc.max_epochs,
            batch_size=rc.batch_size,
            verbose=self.verbose,
            shuffle=True,
            validation_data=rc.validation_data,
            callbacks=[tensorboard]
        )
        self.trained_models[rc.run_name] = rc.model
        self.save_weights(rc.model, rc.run_name)
        return compute_metrics(rc.model, rc.data, rc.batch_size, rc.run_name)

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
