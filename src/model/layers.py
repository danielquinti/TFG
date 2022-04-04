import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers
from tensorflow.keras.utils import plot_model
from src.model import regularizers


def lstm_model(
        inputs,
        layer_config: list,
        regularizer
):
    x = layers.Dense(40, activation='relu')(inputs)
    for layer in layer_config[:-1]:
        x = layers.LSTM(
            layer["units"],
            activation='relu',
            return_sequences=True,
            input_shape=inputs.shape[1:],
            kernel_regularizer=regularizer
        )(x)
    x = layers.LSTM(
        layer_config[-1]["units"],
        activation='relu',
        input_shape=inputs.shape[1:],
        kernel_regularizer=regularizer
    )(x)
    return x


def ffwd_model(inputs):
    x = layers.Flatten()(inputs)
    x = layers.Dense(40, activation='relu')(x)
    x = layers.Dense(30, activation='relu')(x)
    x = layers.Dense(20, activation='relu')(x)
    return x


def get_model(
        config: dict,
        input_shape,
        number_of_classes: dict,
):
    inputs = keras.layers.Input(shape=input_shape)
    try:
        out_regularizer = regularizers.get_regularizer(config["out_regularizer"])
    except KeyError:
        out_regularizer = None
    try:
        in_regularizer = regularizers.get_regularizer(config["in_regularizer"])
    except KeyError:
        in_regularizer = None
    x = {
        "ffwd": ffwd_model(inputs),
        "lstm": lstm_model(
            inputs,
            config["layers"],
            in_regularizer
        )
    }[config["name"]]
    outputs = []
    for feature, n_classes in number_of_classes.items():
        outputs.append(
            layers.Dense(
                n_classes,
                activation='softmax',
                name=feature,
                kernel_regularizer=out_regularizer
            )(x)
        )
    model = keras.Model(inputs=inputs, outputs=outputs)
    plot_model(model, to_file='train_model.png')

    return keras.Model(inputs=inputs, outputs=outputs)
