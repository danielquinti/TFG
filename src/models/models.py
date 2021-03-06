from tensorflow.keras import layers
import tensorflow as tf
import tensorflow.keras as keras
from models import regularizers

def lstm_model(
        input_shape,
        number_of_classes: dict,
        config: dict
):
    embedding_size = config["embedding_size"]
    inputs = layers.Input(shape=input_shape)
    semitone, octave, dur_log, dotted = tf.unstack(inputs, axis=-1)
    semitone = keras.layers.Embedding(13, embedding_size)(semitone)
    octave = keras.layers.Embedding(11, embedding_size)(octave)
    dur_log = keras.layers.Embedding(7, embedding_size)(dur_log)
    dotted = keras.layers.Embedding(2, embedding_size)(dotted)
    e_inputs = keras.layers.Concatenate()([semitone, octave, dur_log, dotted])
    try:
        out_regularizer = regularizers.get_regularizer(config["out_regularizer"])
    except KeyError:
        out_regularizer = None
    try:
        in_regularizer = regularizers.get_regularizer(config["in_regularizer"])
    except KeyError:
        in_regularizer = None
    x = e_inputs
    for layer in config["layers"][:-1]:
        x = layers.LSTM(
            layer["units"],
            activation=config["activation"],
            return_sequences=True,
            input_shape=inputs.shape[1:],
            kernel_regularizer=in_regularizer
        )(x)
    x = layers.LSTM(
        config["layers"][-1]["units"],
        activation=config["activation"],
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
    embedding_size = config["embedding_size"]
    inputs = layers.Input(shape=input_shape)
    semitone, octave, dur_log, dotted = tf.unstack(inputs, axis=-1)
    semitone = keras.layers.Embedding(13, embedding_size)(semitone)
    octave = keras.layers.Embedding(11, embedding_size)(octave)
    dur_log = keras.layers.Embedding(7, embedding_size)(dur_log)
    dotted = keras.layers.Embedding(2, embedding_size)(dotted)
    e_inputs = keras.layers.Concatenate()([semitone, octave, dur_log, dotted])

    x = layers.Flatten()(e_inputs)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dense(32, activation='relu')(x)
    outputs = []
    for feature, n_classes in number_of_classes.items():
        outputs.append(
            layers.Dense(
                n_classes,
                activation='softmax',
                name=feature
            )
            (x)
        )
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model


def last_model(input_shape, number_of_classes: dict, config: dict):
    inputs = layers.Input(shape=input_shape, dtype=tf.uint8)
    x= inputs[:,-1,:]
    feature_indices = {
        "semitone": 0,
        "octave": 1,
        "dur_log": 2,
        "dotted": 3
    }

    outputs = [
        layers.Layer(
            trainable=False,
            name=name
        )
        (
            tf.one_hot(
                x[:,feature_indices[name]],
                depth=value
            )
        ) for name, value in number_of_classes.items()]
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model


def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    # Normalization and Attention
    x = layers.LayerNormalization(epsilon=1e-6)(inputs)
    x = layers.MultiHeadAttention(
        key_dim=head_size, num_heads=num_heads, dropout=dropout
    )(x, x)
    x = layers.Dropout(dropout)(x)
    res = x + inputs

    # Feed Forward Part
    x = layers.LayerNormalization(epsilon=1e-6)(res)
    x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
    return x + res


def transformer_model(
    input_shape,
    number_of_classes: dict,
    config: dict,
    head_size=256,
    num_heads=4,
    ff_dim=4,
    num_transformer_blocks=4,
    mlp_units=[128],
    mlp_dropout=0.4,
    dropout=0.25,
):
    embedding_size = config["embedding_size"]
    inputs = keras.Input(shape=input_shape)
    semitone, octave, dur_log, dotted = tf.unstack(inputs, axis=-1)
    semitone = keras.layers.Embedding(13, embedding_size)(semitone)
    octave = keras.layers.Embedding(11, embedding_size)(octave)
    dur_log = keras.layers.Embedding(7, embedding_size)(dur_log)
    dotted = keras.layers.Embedding(2, embedding_size)(dotted)
    e_inputs = keras.layers.Concatenate()([semitone, octave, dur_log, dotted])
    x = e_inputs
    for _ in range(num_transformer_blocks):
        x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)

    x = layers.GlobalAveragePooling1D(data_format="channels_first")(x)
    for dim in mlp_units:
        x = layers.Dense(dim, activation="relu")(x)
        x = layers.Dropout(mlp_dropout)(x)
    outputs = []
    for feature, n_classes in number_of_classes.items():
        outputs.append(
            layers.Dense(
                n_classes,
                activation='softmax',
                name=feature
            )
            (x)
        )
    return keras.Model(inputs, outputs)


def get_model(
        config: dict,
        input_shape,
        number_of_classes: dict,
):

    models = {
        "last": last_model,
        "ffwd": ffwd_model,
        "lstm": lstm_model,
        "trans": transformer_model
    }
    return models[config["name"]](input_shape, number_of_classes, config)
