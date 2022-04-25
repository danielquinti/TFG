import os

import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.utils import plot_model

from models import optimizers, models, metrics, losses
from preprocessing import dataset


class MyModel:
    def __init__(self, config: dict, output_path, verbose):
        self.output_path = output_path
        self.verbose = verbose
        self.run_name = config["run_name"]
        self.optimizer = optimizers.get_optimizer(config["optimizer"])
        self.batch_size = config["batch_size"]
        self.max_epochs = config["max_epochs"]
        self.input_beats = config["input_beats"]
        self.data = dataset.Dataset(self.input_beats, self.batch_size, self.output_path, *config["encodings"],
                                    config["outputs"])
        self.model: keras.Model = models.get_model(
            config["model"],
            self.data.input_shape,
            self.data.number_of_classes
        )
        outputs = config["outputs"]
        print(self.model.summary())
        # plot_model(
        #     self.model,
        #     to_file=os.path.join(
        #        self.output_path,
        #        f'{self.run_name}_train_model.png'
        #     ),
        #     show_shapes=True, show_layer_names=False
        # )
        self.metrics = {}
        self.losses = {}
        self.loss_weights = {}
        for feature in self.data.number_of_classes.keys():
            self.metrics[feature] = \
                [
                    metrics.get_metric(name, self.data.number_of_classes[feature])
                    for name in outputs[feature]["metrics"]
                ]
            self.losses[feature] = losses.get_loss_function(outputs[feature]["loss"], self.data.class_weights[feature])

            self.loss_weights[feature] = outputs[feature]["loss_weight"]

    def compute_metrics(self):
        evaluation = self.model.evaluate(self.data.test)
        row = [self.run_name] + evaluation
        headers = ["run_name"] + self.model.metrics_names
        return [headers, row]

    def save_weights(self, model, name):
        folder_path = os.path.join(
            self.output_path,
            "weights"
        )
        os.makedirs(folder_path, exist_ok=True)
        weight_filename = os.path.join(
            folder_path,
            f'{name}.h5'
        )
        model.save_weights(weight_filename)

    def run(self):
        self.model.compile(
            optimizer=self.optimizer,
            loss_weights=self.loss_weights,
            loss=self.losses,
            metrics=self.metrics
        )
        log_folder = os.path.join(
            self.output_path,
            "logs"
        )
        tensorboard = keras.callbacks.TensorBoard(
            log_dir=os.path.join(log_folder, self.run_name)
        )

        self.model.fit(
            self.data.train,
            epochs=self.max_epochs,
            verbose=self.verbose,
            validation_data=self.data.test,
            callbacks=[tensorboard]
        )
        self.save_weights(self.model, self.run_name)
        return self.compute_metrics()

    def load_weights(self):
        weight_filename = os.path.join(
            "weights",
            f'{self.run_name}.h5'
        )
        self.model.load_weights(weight_filename)

    def predict(self, data, length):
        music=data.as_numpy_iterator().next()[0,:,:].reshape(1,30,4)
        inp= np.copy(music)
        for i in range(length):
            prediction=self.model.predict(inp)
            beat = np.array([np.argmax(x[0,:]) for x in prediction]).reshape(1,1,4)
            inp = np.concatenate((inp[:,i+1:,:], beat), axis=1)
            music = np.concatenate((music, beat), axis=1)
        return music.reshape(-1,4)
