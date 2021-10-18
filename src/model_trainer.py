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
        self.trained_models = {}

    def compile_and_fit(self,
                        model,
                        name,
                        loss_names: dict,
                        metric_names: dict,
                        loss_weights,
                        optimizer_name,
                        batch_size,
                        max_epochs,
                        verbose
                        ):
        folder_name = f'{name}_{loss_names["notes"]}_{metric_names["notes"]}_lw{loss_weights["notes"]},{loss_weights["duration"]}_bs{batch_size}_e{max_epochs}'
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
                folder_name
            )
        )

        return model, history

    def run(self):

        num_n_classes = len(self.dataset.note_classes)
        num_d_classes = len(self.dataset.duration_classes)
        for config in self.model_configs:

            mc = ModelConfiguration(config)
            model = available_models[mc.model_name](num_n_classes, num_d_classes)

            self.compile_and_fit(
                model,
                mc.model_name,
                mc.loss_function_names,
                mc.metric_names,
                mc.loss_weights,
                mc.optimizer_name,
                mc.batch_size,
                mc.max_epochs,
                self.verbose,
            )

    def load_models(self,
                    input_path,
                    selected_models):
        for selected_name in selected_models:
            for av_name, model_builder in available_models.items():
                if selected_name == av_name:
                    os.chdir(input_path)
                    self.trained_models[av_name] = load_model(av_name)
                    os.chdir("..")


class ModelConfiguration:
    def __init__(self, data: dict):
        self.model_name = data["model_name"]
        self.loss_function_names: dict = data["loss_function_names"]
        self.metric_names: dict = data["metric_names"]
        self.loss_weights: dict = data["loss_weights"]
        self.optimizer_name = data["optimizer_name"]
        self.batch_size = data["batch_size"]
        self.max_epochs = data["max_epochs"]
