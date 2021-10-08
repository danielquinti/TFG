#!/usr/bin/python3
import itertools

import tensorflow as tf
from keras.models import load_model
from sklearn.metrics import confusion_matrix
from CSV2Dataset import *
from src.available_models import available_models
import json

class ModelTrainer:
    def __init__(self, dataset, create=True, save=True):
        config = ConfigParser()
        config.read('.\config.ini')
        config_group = "ModelTrainer"
        self.selected_models = [value for value in (config.get(config_group, "selected_models").split(","))]
        self.batch_size = int(config.get(config_group, "batch_size"))
        self.max_epochs = int(config.get(config_group, "max_epochs"))
        self.loss_weights = [int(value) for value in (config.get(config_group, "loss_weights").split(","))]
        self.output_path = config.get(config_group, "output_path")
        self.models = {}
        self.train(dataset, create, save)

    def compile_and_train(self, model, dataset):
        model.compile(
            loss=dict(notes=tf.keras.losses.CategoricalCrossentropy(),
                      duration=tf.keras.losses.CategoricalCrossentropy()),
            optimizer='adam',
            metrics=dict(notes='accuracy', duration='accuracy'),
            loss_weights=dict(notes=self.loss_weights[0], duration=self.loss_weights[1])
        )

        history = model.fit(x=dataset.train_inputs,
                            y={'notes': dataset.train_labels["notes"], 'duration': dataset.train_labels["duration"]},
                            epochs=self.max_epochs,
                            batch_size=self.batch_size,
                            verbose=2,
                            shuffle=True,
                            validation_data=(dataset.test_inputs, dataset.test_labels.values())
                            )
        return model, history

    def train(self, dataset, create, save):
        for selected_name in self.selected_models:
            for av_name, model_builder in available_models.items():
                if selected_name == av_name:
                    if create:
                        model = model_builder()
                        model, history = self.compile_and_train(model, dataset)
                        if save:
                            self.models[av_name] = model
                            model.save(os.path.join(self.output_path, f'av_name.h5'))
                            with open(os.path.join(self.output_path, f'{av_name}_history.json'), 'w') as fp:
                                json.dump(history.history, fp)
                    else:
                        self.models[av_name] = load_model(os.path.join(self.output_path, f'{av_name}.h5'))
