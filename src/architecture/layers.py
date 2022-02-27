import tensorflow.keras as keras
from tensorflow.keras import layers
import regularizers

def lstm_model(
        inputs,
        layer_config: list,
        input_beats,
        input_range,
        regularizer
):
    x=inputs
    for layer in layer_config[:-1]:
        x = layers.LSTM(
            layer["units"],
            activation='relu',
            return_sequences=True,
            input_shape=(input_beats, input_range),
            kernel_regularizer=regularizer
        )(x)
    x = layers.LSTM(
        layer_config[-1]["units"],
        activation='relu',
        input_shape=(input_beats, input_range),
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
        input_range,
        number_of_classes: dict,
        input_beats,
        label_beats,
        active_features: dict,
):
    try:
        out_regularizer = regularizers.get_regularizer(config["out_regularizer"])
    except KeyError:
        out_regularizer = None
    try:
        in_regularizer = regularizers.get_regularizer(config["in_regularizer"])
    except KeyError:
        in_regularizer = None
    inputs = layers.Input((input_beats, input_range))
    x = {
        "ffwd": ffwd_model(inputs),
        "lstm": lstm_model(
            inputs,
            config["layers"],
            input_range,
            input_beats,
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
