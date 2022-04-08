from tensorflow.keras import layers
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.python.keras.utils.vis_utils import plot_model

from models import regularizers

def lstm_model(
        input_shape,
        number_of_classes: dict,
        config: list
):
    inputs = layers.Input(shape=input_shape)
    semitone, octave, dur_log, dotted = tf.unstack(inputs, axis=-1)
    semitone = keras.layers.Embedding(13, 13)(semitone)
    octave = keras.layers.Embedding(10, 10)(octave)
    dur_log = keras.layers.Embedding(7, 7)(dur_log)
    dotted = keras.layers.Embedding(2, 2)(dotted)
    e_inputs = keras.layers.Concatenate()([semitone, octave, dur_log, dotted])
    try:
        out_regularizer = regularizers.get_regularizer(config["out_regularizer"])
    except KeyError:
        out_regularizer = None
    try:
        in_regularizer = regularizers.get_regularizer(config["in_regularizer"])
    except KeyError:
        in_regularizer = None
    x = inputs
    for layer in config["layers"][:-1]:
        x = layers.LSTM(
            layer["units"],
            activation='relu',
            return_sequences=True,
            input_shape=inputs.shape[1:],
            kernel_regularizer=in_regularizer
        )(x)
    x = layers.LSTM(
        config["layers"][-1]["units"],
        activation='relu',
        input_shape=inputs.shape[1:],
        kernel_regularizer=in_regularizer
    )(x)
    outputs = []
    for feature, n_classes in number_of_classes.items():
        outputs.append(
            layers.Dense(
                n_classes,
                activation='softmax',
                name=feature,
                kernel_regularizer=out_regularizer
            )
            (x)
        )
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model


def ffwd_model(input_shape, number_of_classes: dict, config: dict):
    inputs = layers.Input(shape=input_shape)
    semitone, octave, dur_log, dotted = tf.unstack(inputs, axis=-1)
    semitone = keras.layers.Embedding(13, 13)(semitone)
    octave = keras.layers.Embedding(10, 10)(octave)
    dur_log = keras.layers.Embedding(7, 7)(dur_log)
    dotted = keras.layers.Embedding(2, 2)(dotted)
    e_inputs = keras.layers.Concatenate()([semitone, octave, dur_log, dotted])
    try:
        out_regularizer = regularizers.get_regularizer(config["out_regularizer"])
    except KeyError:
        out_regularizer = None
    try:
        in_regularizer = regularizers.get_regularizer(config["in_regularizer"])
    except KeyError:
        in_regularizer = None

    x = layers.Flatten()(inputs)
    x = layers.Dense(40, activation='relu')(x)
    x = layers.Dense(30, activation='relu')(x)
    x = layers.Dense(20, activation='relu')(x)
    outputs = []
    for feature, n_classes in number_of_classes.items():
        outputs.append(
            layers.Dense(
                n_classes,
                activation='softmax',
                name=feature,
                kernel_regularizer=out_regularizer
            )
            (x)
        )
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model


def last_model(input_shape, number_of_classes: dict, config: dict):
    inputs = layers.Input(shape=input_shape, dtype=tf.uint8)
    window_beats = tf.unstack(inputs, axis=1)
    raw_outputs = tf.unstack(window_beats[-1], axis=-1)
    oh_outputs = [tf.one_hot(raw_output, depth=n_classes) for raw_output, n_classes in zip(raw_outputs, number_of_classes.values())]
    outputs = [
        layers.Layer(
            trainable=False,
            name=name
        )(data) for name, data in zip(number_of_classes.keys(), oh_outputs)]
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model


def get_model(
        config: dict,
        input_shape,
        number_of_classes: dict,
):

    models = {
        "last": last_model,
        "ffwd": ffwd_model,
        "lstm": last_model
        # "trans": transformer_model
    }
    return models[config["name"]](input_shape, number_of_classes, config)
