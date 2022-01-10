#!/usr/bin/python3
import csv
import json
import os

from tensorflow.python import keras

from data_processing import dataset_manager, dataset
from model import losses, metrics, models


def compute_metrics(model, data, mc):
    evaluation = model.evaluate(
        data.test.inputs,
        [
            data.test.labels.notes,
            data.test.labels.duration
        ],
        mc.batch_size
    )
    row = [mc.folder_name] + evaluation
    headers = ["model"] + model.metrics_names
    return row, headers


class ModelConfig:
    def __init__(self, config):
        self.model_name = config["model_name"]
        self.loss_function_names: dict = config["loss_function_names"]
        self.metric_names: dict = config["metric_names"]
        self.loss_weights: dict = config["loss_weights"]
        self.optimizer_name = config["optimizer_name"]
        self.batch_size = config["batch_size"]
        self.max_epochs = config["max_epochs"]
        self.input_beats = config["input_beats"]
        self.label_beats = config["label_beats"]
        self.folder_name = f'{self.model_name}_{self.loss_function_names["notes"]}_opt_{self.optimizer_name}_' + \
                           f'lw({self.loss_weights["notes"]},{self.loss_weights["duration"]})_bs{self.batch_size}_e' + \
                           f'{self.max_epochs}_ds({self.input_beats},{self.label_beats})'


class ModelTrainer:
    def __init__(self):
        fp = open(
            os.path.join(
                "src",
                "config",
                "train_config.json"
            )
        )
        params = json.load(fp)

        self.output_path = os.path.join(*(params["output_path"].split("\\")[0].split("/")))
        self.verbose = params["verbose"]
        self.model_configs: list = params["model_configs"]
        self.dataset_manager = dataset_manager.DatasetManager()
        self.trained_models = {}

    def compile_model(
            self,
            model,
            data,
            loss_names,
            optimizer_name,
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
            optimizer=optimizer_name,
            loss_weights=loss_weights,
            loss=metr_dict,
            metrics=loss_dict
        )

    def fit_model(
            self,
            model,
            data,
            max_epochs,
            batch_size,
            folder_name
    ):
        route = os.path.join(
            "logs",
            folder_name
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
            verbose=self.verbose,
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

    def build_model(
            self,
            mc: ModelConfig,
            data: dataset.Dataset
    ):
        model = models.get_model(
            mc.model_name,
            data.n_classes,
            data.d_classes,
            mc.input_beats,
            mc.label_beats
        )

        self.compile_model(
            model,
            data,
            mc.loss_function_names,
            mc.optimizer_name,
            mc.loss_weights,
            mc.metric_names
        )

        self.fit_model(
            model,
            data,
            mc.max_epochs,
            mc.batch_size,
            mc.folder_name
        )
        return model

    def save_weights(self, model, name):
        folder_path = os.path.join(
            self.output_path,
            f'{name}.h5'
        )
        if not os.path.exists(self.output_path):
            os.mkdir(self.output_path)
        model.save_weights(folder_path)

    def run(self):
        rows = []
        for config in self.model_configs:
            mc = ModelConfig(config)
            data = self.dataset_manager.get_dataset(mc.input_beats, mc.label_beats)
            model = self.build_model(mc, data)
            self.trained_models[mc.folder_name] = model
            self.save_weights(model, mc.folder_name)
            row, header = compute_metrics(model, data, mc)
            if not rows:
                rows.append(header)
            rows.append(row)
        with open(
            os.path.join(
                self.output_path,
                "metrics.csv"
            ),
            'w',
        ) as csv_file:
            writer = csv.writer(csv_file)
            writer.writerows(rows)

    def load_models(self):
        for config in self.model_configs:
            mc = ModelConfig(config)
            data = self.dataset_manager.get_dataset(mc.input_beats, mc.label_beats)
            self.load_model(mc, data)

    def load_model(self,
                   mc: ModelConfig,
                   data: dataset.Dataset):
        model = models[mc.model_name](
            data.n_classes,
            data.d_classes,
            mc.input_beats,
            mc.label_beats
        )
        model.load_weights(
            os.path.join(
                self.output_path,
                f'{mc.folder_name}.h5'
            )
        )
        self.trained_models[mc.folder_name] = (model, data)
