from tensorflow import keras

from src.data_management import dataset

from src.model import layers, metrics, optimizers, losses


class RunConfig:
    def __init__(self, config):
        self.run_name = config["run_name"]
        self.optimizer = optimizers.get_optimizer(config["optimizer"])
        self.batch_size = config["batch_size"]
        self.max_epochs = config["max_epochs"]
        self.input_beats = config["input_beats"]
        raw_loss_weights = config["loss_weights"]
        active_features = {
            feature: raw_loss_weights[feature] > 0 for feature in raw_loss_weights.keys()
        }
        self.data = dataset.Dataset(self.input_beats, self.batch_size, active_features)
        self.model: keras.Model = layers.get_model(
            config["model"],
            self.data.input_shape,
            self.data.number_of_classes
        )

        metric_name = config["metric_name"]
        loss_name = config["loss_function_name"]
        self.metrics = {}
        self.losses = {}
        self.loss_weights = {}
        for feature in self.data.number_of_classes.keys():
            self.metrics[feature] = \
                [
                    metrics.get_metric(name, self.data.number_of_classes[feature]) for name in metric_name
                ]
            self.losses[feature] = losses.get_loss_function(
                loss_name
            )

            self.loss_weights[feature] = 1. / len(self.data.number_of_classes.keys())
        print()
