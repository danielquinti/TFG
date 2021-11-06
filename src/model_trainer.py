#!/usr/bin/python3
import csv

import keras
from keras.models import load_model
from data_processing.dataset_manager import *
from model.available_models import available_models
from model.metrics import *
from model.losses import *
import json
from time import time
from tensorflow.python.keras.callbacks import TensorBoard


def alt_measurement(raw_true, raw_pred):
    true = np.argmax(raw_true, axis=1)
    pred = np.argmax(raw_pred, axis=1)
    data = confusion_matrix(
        true,
        pred
    )
    diag = np.diag(data)
    true_counts = np.sum(data, axis=1)

    # avoid division by 0
    recalls = np.divide(diag, true_counts, out=np.zeros_like(diag, dtype='float'), where=true_counts != 0)
    # compute mean discarding labels with no counts
    metric_cm = recalls.sum() / np.sign(true_counts).sum()
    return [metric_cm]


def compute_metrics(model, dataset, mc):
    evaluation = model.evaluate(
        dataset.test.inputs,
        [
            dataset.test.labels.notes,
            dataset.test.labels.duration
        ],
        mc.batch_size
    )

    notes_pred, duration_pred = model.predict(
        dataset.test.inputs
    )
    notes = alt_measurement(dataset.test.labels.notes, notes_pred)
    duration = alt_measurement(dataset.test.labels.duration, duration_pred)
    row = [mc.folder_name] + \
        evaluation[:-1] + \
        notes + \
        [evaluation[-1]] + \
        duration
    headers = ["model"] + \
        model.metrics_names[:-1] + \
        ["notes-cm"] + \
        [model.metrics_names[-1]] + \
        ["duration-cm"]
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
        self.folder_name = f'{self.model_name}_{self.loss_function_names["notes"]}_lw' + \
                           f'({self.loss_weights["notes"]},{self.loss_weights["duration"]})_bs{self.batch_size}_e' + \
                           f'{self.max_epochs}_ds({self.input_beats},{self.label_beats})'


class ModelTrainer:
    def __init__(self):
        fp = open(
            os.path.join(
                "src",
                "train_config.json"
            )
        )
        params = json.load(fp)

        self.output_path = os.path.join(*(params["output_path"].split("\\")[0].split("/")))
        self.verbose = params["verbose"]
        self.model_configs: list = params["model_configs"]
        self.dataset_manager = DatasetManager()
        self.trained_models = {}

    def compile_model(
            self,
            model,
            dataset,
            loss_names,
            optimizer_name,
            loss_weights,
            metric_names
    ):
        metrics = {
            "notes":
                [
                    get_metric(name, dataset.n_classes) for name in metric_names["notes"]
                ],
            "duration":
                [
                    get_metric(name, dataset.d_classes) for name in metric_names["duration"]
                ],
        }
        losses = {
            "notes": get_loss_function(
                loss_names["notes"],
                dataset.notes_weights
            ),
            "duration": get_loss_function(
                loss_names["duration"],
                dataset.duration_weights
            ),
        }
        model.compile(
            optimizer=optimizer_name,
            loss_weights=loss_weights,
            loss=losses,
            metrics=metrics
        )

    def fit_model(
            self,
            model,
            dataset,
            max_epochs,
            batch_size,
            folder_name
    ):
        route = os.path.join(
            "logs",
            folder_name
        )

        tensorboard = TensorBoard(log_dir=route)

        model.fit(
            x=dataset.train.inputs,
            y={
                'notes': dataset.train.labels.notes,
                'duration': dataset.train.labels.duration
            },
            epochs=max_epochs,
            batch_size=batch_size,
            verbose=self.verbose,
            shuffle=True,
            validation_data=(
                dataset.test.inputs,
                [
                    dataset.test.labels.notes,
                    dataset.test.labels.duration
                ]
            ),
            callbacks=[tensorboard]
        )

    def build_model(
            self,
            mc: ModelConfig,
            dataset: Dataset
    ):
        model = available_models[mc.model_name](
            dataset.n_classes,
            dataset.d_classes,
            mc.input_beats,
            mc.label_beats
        )

        self.compile_model(
            model,
            dataset,
            mc.loss_function_names,
            mc.optimizer_name,
            mc.loss_weights,
            mc.metric_names
        )

        self.fit_model(
            model,
            dataset,
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
            dataset = self.dataset_manager.get_dataset(mc.input_beats, mc.label_beats)
            model = self.build_model(mc, dataset)
            self.trained_models[mc.folder_name] = model
            self.save_weights(model, mc.folder_name)
            row, header = compute_metrics(model, dataset, mc)
            if not rows:
                rows.append(header)
            rows.append(row)
        with open(
                os.path.join(
                    self.output_path,
                    "metrics.csv"
                ),
                'w',
        ) as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(rows)

    def load_models(self):
        for config in self.model_configs:
            mc = ModelConfig(config)
            dataset = self.dataset_manager.get_dataset(mc.input_beats, mc.label_beats)
            self.load_model(mc, dataset)

    def load_model(self,
                   mc: ModelConfig,
                   dataset: Dataset):
        model = available_models[mc.model_name](
            dataset.n_classes,
            dataset.d_classes,
            mc.input_beats,
            mc.label_beats
        )
        model.load_weights(
            os.path.join(
                self.output_path,
                f'{mc.folder_name}.h5'
            )
        )
        self.trained_models[mc.folder_name] = (model, dataset)
