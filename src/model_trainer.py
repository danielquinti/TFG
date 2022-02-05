#!/usr/bin/python3
import csv
import json
import os
from datetime import datetime

from tensorflow.python import keras

from architecture import losses, metrics, layers, regularizers, optimizers
from data_processing import dataset_manager, dataset


def compute_metrics(model, data, mc):
    evaluation = model.evaluate(
        data.test.inputs,
        [
            data.test.labels.notes,
            data.test.labels.duration
        ],
        mc.batch_size
    )
    row = [mc.run_name] + evaluation
    headers = ["run_name"] + model.metrics_names
    return row, headers


def compile_model(
        model,
        data,
        loss_names,
        optimizer,
        loss_weights,
        metric_names
):
    metr_dict = {
        "notes":
            [
                metrics.get_metric(name, data.n_classes) for name in metric_names["notes"]
            ],
        "duration":
            [
                metrics.get_metric(name, data.d_classes) for name in metric_names["duration"]
            ],
    }
    loss_dict = {
        "notes": losses.get_loss_function(
            loss_names["notes"],
            data.notes_weights
        ),
        "duration": losses.get_loss_function(
            loss_names["duration"],
            data.duration_weights
        ),
    }
    model.compile(
        optimizer=optimizer,
        loss_weights=loss_weights,
        loss=loss_dict,
        metrics=metr_dict
    )


def fit_model(
        model,
        data,
        max_epochs,
        batch_size,
        output_path,
        run_name,
        verbose
):
    route = os.path.join(
        output_path,
        "logs",
        run_name
    )

    tensorboard = keras.callbacks.TensorBoard(log_dir=route)

    model.fit(
        x=data.train.inputs,
        y={
            'notes': data.train.labels.notes,
            'duration': data.train.labels.duration
        },
        epochs=max_epochs,
        batch_size=batch_size,
        verbose=verbose,
        shuffle=True,
        validation_data=(
            data.test.inputs,
            [
                data.test.labels.notes,
                data.test.labels.duration
            ]
        ),
        callbacks=[tensorboard]
    )


class RunConfig:
    def __init__(self, config):
        self.run_name = config["run_name"]
        self.model_name = config["model_name"]
        self.loss_function_names: dict = config["loss_function_names"]
        self.metric_names: dict = config["metric_names"]
        self.loss_weights: dict = config["loss_weights"]
        try:
            self.regularizer = regularizers.get_regularizer(config["regularizer"])
        except KeyError:
            self.regularizer = None
        self.optimizer = optimizers.get_optimizer(config["optimizer"])
        self.batch_size = config["batch_size"]
        self.max_epochs = config["max_epochs"]
        self.input_beats = config["input_beats"]
        self.label_beats = config["label_beats"]


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

    def build_model(
            self,
            mc: RunConfig,
            data: dataset.Dataset,
    ):
        model = layers.get_model(
            mc.model_name,
            data.n_classes,
            data.d_classes,
            mc.input_beats,
            mc.label_beats,
            mc.loss_weights,
            mc.regularizer
        )

        compile_model(
            model,
            data,
            mc.loss_function_names,
            mc.optimizer,
            mc.loss_weights,
            mc.metric_names
        )

        fit_model(
            model,
            data,

            mc.max_epochs,
            mc.batch_size,
            self.output_path,
            mc.run_name,
            self.verbose
        )
        return model

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

    def run_all(self):
        rows = []
        for config in self.model_configs:
            mc = RunConfig(config)
            data = self.dataset_manager.get_dataset(mc.input_beats, mc.label_beats)
            model = self.build_model(mc, data)
            self.trained_models[mc.run_name] = model
            self.save_weights(model, mc.run_name)
            row, header = compute_metrics(model, data, mc)
            if not rows:
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
