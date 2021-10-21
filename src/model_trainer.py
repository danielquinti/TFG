#!/usr/bin/python3

from keras.models import load_model
from data_processing.dataset_manager import *
from model.available_models import available_models
from model.metrics import *
from model.losses import *
import json
from time import time
from tensorflow.python.keras.callbacks import TensorBoard


class ModelTrainer:
    def __init__(self, dataset: Dataset):
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
        self.dataset = dataset
        self.num_n_classes = self.dataset.train.labels.notes.shape[1]
        self.num_d_classes = self.dataset.train.labels.duration.shape[1]
        self.trained_models = {}

    def compile_and_fit(self,
                        model,
                        folder_name,
                        loss_names: dict,
                        metric_names: dict,
                        loss_weights,
                        optimizer_name,
                        batch_size,
                        max_epochs,
                        verbose
                        ):
        losses = {
            "notes": get_loss_function(loss_names["notes"], self.dataset.weights["train_notes"]),
            "duration": get_loss_function(loss_names["duration"], self.dataset.weights["train_duration"])
        }
        model.compile(
            loss=losses,
            optimizer=optimizer_name,
            metrics={
                "notes": metrics[metric_names["notes"]],
                "duration": metrics[metric_names["duration"]]
            },
            loss_weights=loss_weights
        )

        route = os.path.join(
            "logs",
            folder_name
        )

        tensorboard = TensorBoard(log_dir=route)

        history = model.fit(
            x=self.dataset.train.inputs,
            y={
                'notes': self.dataset.train.labels.notes,
                'duration': self.dataset.train.labels.duration
            },
            epochs=max_epochs,
            batch_size=batch_size,
            verbose=verbose,
            shuffle=True,
            validation_data=(
                self.dataset.test.inputs,
                [
                    self.dataset.test.labels.notes,
                    self.dataset.test.labels.duration
                ]
            ),
            callbacks=[tensorboard]
        )

        model.save(
            os.path.join(
                self.output_path,
                f'{folder_name}.h5'
            )
        )

        return model, history

    def train_models(self):
        for config in self.model_configs:

            mc = ModelConfiguration(config)
            model = available_models[mc.model_name](self.num_n_classes, self.num_d_classes)
            self.compile_and_fit(
                model,
                mc.folder_name,
                mc.loss_function_names,
                mc.metric_names,
                mc.loss_weights,
                mc.optimizer_name,
                mc.batch_size,
                mc.max_epochs,
                self.verbose,
            )
            self.trained_models[mc.folder_name]=model

    def load_models(self):

        for config in self.model_configs:
            mc = ModelConfiguration(config)
            model = available_models[mc.model_name](self.num_n_classes, self.num_d_classes)
            model.load_weights(
                os.path.join(self.output_path,
                             f'{mc.folder_name}.h5'
                             )
            )
            self.trained_models[mc.folder_name]=model
class ModelConfiguration:
    def __init__(self, data: dict):
        self.model_name = data["model_name"]
        self.loss_function_names: dict = data["loss_function_names"]
        self.metric_names: dict = data["metric_names"]
        self.loss_weights: dict = data["loss_weights"]
        self.optimizer_name = data["optimizer_name"]
        self.batch_size = data["batch_size"]
        self.max_epochs = data["max_epochs"]
        self.folder_name = f'{self.model_name}_{self.loss_function_names["notes"]}_{self.metric_names["notes"]}_lw{self.loss_weights["notes"]},{self.loss_weights["duration"]}_bs{self.batch_size}_e{self.max_epochs}'
