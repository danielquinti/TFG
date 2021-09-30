#!/usr/bin/python3
import itertools

import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras as keras
from sklearn.metrics import confusion_matrix
from tensorflow.keras import layers
import time
from DatasetFromCSV import *
from src.ConfigManager import ConfigManager
from src.models import available_models









class ModelBenchmark():
    def __init__(self, config, dataset):
        self.config = config
        self.dataset = dataset

    def compile_and_train(self, model, dataset):
        model.compile(
            loss=dict(notes=tf.keras.losses.CategoricalCrossentropy(), duration=tf.keras.losses.MeanSquaredError()),
            optimizer='adam',
            metrics=dict(notes="accuracy", duration=keras.metrics.MeanAbsoluteError()),
            loss_weights=dict(notes=self.config.loss_weights[0], duration=self.config.loss_weights[1])
        )

        history = model.fit(x=dataset.train_inputs,
                            y={'notes': dataset.train_labels["notes"], 'duration': dataset.train_labels["duration"]},
                            epochs=self.config.max_epochs,
                            batch_size=self.config.batch_size,
                            verbose=2,
                            shuffle=True,
                            validation_split=self.config.validation_split
                            )
        return model, history

    def save_cms(self, model, dataset, name):
        predictions = model.predict(dataset.test_inputs, verbose=0)

        obtained = np.argmax(predictions[0], axis=1).flatten()
        expected = np.argmax(dataset.test_labels["notes"], axis=1).flatten()
        data = np.vstack([obtained,expected])
        np.savetxt((self.config.output_dir + name + "_notes_cm.csv"), data, fmt='%i')

        obtained = np.round(np.log2(predictions[1])).flatten()
        expected = np.round(np.log2(dataset.test_labels["duration"])).flatten()
        data = np.vstack([obtained,expected])
        np.savetxt((self.config.output_dir + name + "_duration_cm.csv"), data, fmt='%i')

    def execute(self):
        for name, model_builder in available_models.items():
            if name in self.config.selected_models:
                model = model_builder()
                model, history = self.compile_and_train(model, self.dataset)
                self.save_cms(model, self.dataset, name)
                for key, value in history.history.items():
                    np.savetxt(self.config.output_dir + name + "_" + key + ".txt", value)

