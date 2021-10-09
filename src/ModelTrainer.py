#!/usr/bin/python3

import tensorflow as tf
from keras.models import load_model
from CSV2Dataset import *
from src.available_models import available_models
import json


def compile_and_train(model,
                      dataset,
                      batch_size,
                      max_epochs,
                      loss_weights
                      ):
    model.compile(
        loss=dict(notes=tf.keras.losses.CategoricalCrossentropy(),
                  duration=tf.keras.losses.CategoricalCrossentropy()),
        optimizer='adam',
        metrics=dict(notes='accuracy', duration='accuracy'),
        loss_weights=dict(notes=loss_weights[0], duration=loss_weights[1])
    )

    history = model.fit(x=dataset.train_inputs,
                        y={
                            'notes': dataset.train_labels["notes"],
                            'duration': dataset.train_labels["duration"]
                        },
                        epochs=max_epochs,
                        batch_size=batch_size,
                        verbose=2,
                        shuffle=True,
                        validation_data=(dataset.test_inputs, dataset.test_labels.values())
                        )
    return model, history


class ModelTrainer:
    def __init__(self,
                 dataset,
                 selected_models,
                 batch_size,
                 max_epochs,
                 loss_weights,
                 output_path,
                 create=True,
                 save=True):
        self.trained_models = {}
        for selected_name in selected_models:
            for av_name, model_builder in available_models.items():
                if selected_name == av_name:
                    if create:
                        model = model_builder()
                        model, history = compile_and_train(model,
                                                           dataset,
                                                           batch_size,
                                                           max_epochs,
                                                           loss_weights)
                        if save:
                            self.trained_models[av_name] = model
                            model.save(os.path.join(output_path, f'av_name.h5'))
                            with open(os.path.join(output_path, f'{av_name}_history.json'), 'w') as fp:
                                json.dump(history.history, fp)
                    else:
                        self.trained_models[av_name] = load_model(os.path.join(output_path, f'{av_name}.h5'))
