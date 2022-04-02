import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers
from src.architecture import regularizers


def lstm_model(
        inputs,
        layer_config: list,
        regularizer
):
    x=inputs
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


def preprocess(x):
    x = tf.cast(x, tf.int32)
    window_size = x.shape[1]
    notes = x[:, :, 0]
    notes = tf.reshape(notes, shape=[-1])

    octaves = x[:, :, 1]
    octaves = tf.reshape(octaves, shape=[-1])

    real_notes = octaves * 12 + notes
    oh_notes = layers.experimental.preprocessing.CategoryEncoding(num_tokens=121, output_mode="one_hot")(real_notes)

    duration = tf.cast(x[:, :, 2], tf.int32)
    duration = tf.reshape(duration, shape=[-1])

    dotted = tf.cast(x[:, :, 3], tf.int32)
    dotted = tf.reshape(dotted, shape=[-1])
    real_duration = duration * 2 + dotted
    oh_duration = layers.experimental.preprocessing.CategoryEncoding(num_tokens=14, output_mode="one_hot")(real_duration)
    oh_concat = tf.concat([oh_notes, oh_duration], axis=1)
    return tf.reshape(oh_concat, shape=[-1, window_size, oh_concat.shape[-1]])


def ffwd_model(inputs):
    x = layers.Flatten()(inputs)
    x = layers.Dense(40, activation='relu')(x)
    x = layers.Dense(30, activation='relu')(x)
    x = layers.Dense(20, activation='relu')(x)
    return x

def preprocess_data(data: tf.data.Dataset):
    window_size = data.element_spec[0].shape[0]
    n_features = data.element_spec[0].shape[1]
    inputs = keras.Input(shape=(window_size, n_features), dtype=data.element_spec[0].dtype)
    preprocessed_inputs = preprocess(inputs)
    return preprocessed_inputs, tf.keras.Model(inputs=inputs, outputs=preprocessed_inputs)



def get_model(
        config: dict,
        input_shape,
        number_of_classes : dict,
        active_features: dict,
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
    for feature, is_active in active_features.items():
        if is_active:
            outputs.append(
                layers.Dense(
                    number_of_classes[feature],
                    activation='softmax',
                    name=feature,
                    kernel_regularizer=out_regularizer
                )(x)
            )
    return keras.Model(inputs=inputs, outputs=outputs)
