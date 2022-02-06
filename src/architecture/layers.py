import tensorflow.keras as keras
from tensorflow.keras import layers


def lstm_model(inputs, input_beats, input_range, regularizer):
    x = layers.LSTM(
        # TODO parameterize
        65,
        activation='relu',
        input_shape=(input_beats, input_range),
        kernel_regularizer=regularizer
    )(inputs)
    return x


def ffwd_model(inputs):
    x = layers.Flatten()(inputs)
    x = layers.Dense(40, activation='relu')(x)
    x = layers.Dense(30, activation='relu')(x)
    x = layers.Dense(20, activation='relu')(x)
    return x


def get_model(
        model_name,
        n_classes,
        d_classes,
        input_beats,
        label_beats,
        notes_active,
        duration_active,
        regularizer
):
    input_range = n_classes + d_classes
    inputs = layers.Input((input_beats, input_range))
    x = {
        "ffwd": ffwd_model(inputs),
        "lstm": lstm_model(inputs, n_classes + d_classes, input_beats, regularizer)
    }[model_name]
    outputs=[]
    if notes_active:
        outputs.append(
            layers.Dense(
                n_classes,
                activation='softmax',
                name='notes',
                kernel_regularizer=regularizer
            )(x)
        )
    if duration_active:
        outputs.append(
            layers.Dense(
                d_classes,
                activation='softmax',
                name='duration',
                kernel_regularizer=regularizer
            )(x)
        )
    return keras.Model(inputs=inputs, outputs=outputs)
