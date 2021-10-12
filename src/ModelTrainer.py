#!/usr/bin/python3

import tensorflow as tf
from keras.models import load_model
from src.data_processing.CSV2Dataset import *
from model.available_models import available_models
from model.losses import weighted_cce_n, weighted_cce_d
from model.metrics import mean_ap
import json


def compile_and_train(model,
                      dataset,
                      batch_size,
                      max_epochs,
                      loss_weights
                      ):
    model.compile(
        loss=dict(
            # notes=tf.keras.losses.CategoricalCrossentropy(),
            # duration=tf.keras.losses.CategoricalCrossentropy(),
            notes = weighted_cce_n,
            duration = weighted_cce_d
        ),
        optimizer='adam',
        metrics=dict(notes="accuracy", duration="accuracy"),
        loss_weights=loss_weights
    )

    history = model.fit(x=dataset.train.inputs,
                        y={
                            'notes': dataset.train.labels.notes,
                            'duration': dataset.train.labels.duration
                        },
                        epochs=max_epochs,
                        batch_size=batch_size,
                        verbose=2,
                        shuffle=False,
                        validation_data=(
                            dataset.test.inputs,
                            [
                                dataset.test.labels.notes,
                                dataset.test.labels.duration
                            ]
                        )
                        )
    return model, history


class ModelTrainer:
    def __init__(self):
        self.trained_models = {}

    def train_models(self,
                     dataset,
                     selected_models,
                     batch_size,
                     max_epochs,
                     loss_weights,
                     output_path,
                     save=True):
        for selected_name in selected_models:
            for av_name, model_builder in available_models.items():
                if selected_name == av_name:
                    model = model_builder()
                    model, history = compile_and_train(model,
                                                       dataset,
                                                       batch_size,
                                                       max_epochs,
                                                       loss_weights)
                    if save:
                        self.trained_models[av_name] = model
                        model.save(os.path.join(output_path, av_name))
                        with open(os.path.join(output_path, f'{av_name}_history.json'), 'w') as fp:
                            json.dump(history.history, fp)

    def load_models(self,
                    input_path,
                    selected_models):
        for selected_name in selected_models:
            for av_name, model_builder in available_models.items():
                if selected_name == av_name:
                    os.chdir(input_path)
                    self.trained_models[av_name] = load_model(av_name)
                    os.chdir("..")
