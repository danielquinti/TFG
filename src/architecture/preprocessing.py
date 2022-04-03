import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import plot_model


def preprocess_inputs(x):
    window_size = x.shape[1]
    x=tf.reshape(x, (-1, 4))

    oh_pitch = notes_and_octaves(x)
    oh_duration = duration_and_dotted(x)
    oh_concat = tf.concat([oh_pitch, oh_duration], axis=1)
    return tf.reshape(oh_concat, shape=[-1, window_size, oh_concat.shape[-1]])


def preprocess_notes(x):
    x = tf.cast(x, tf.int32)
    notes = x[:, 0]
    oh_notes = layers.experimental.preprocessing.CategoryEncoding(num_tokens=13, output_mode="one_hot")(notes)
    return oh_notes


def preprocess_octaves(x):
    x = tf.cast(x, tf.int32)
    octaves = x[:, 1]
    oh_octaves = layers.experimental.preprocessing.CategoryEncoding(num_tokens=10, output_mode="one_hot")(octaves)
    return oh_octaves

def notes_and_octaves(x):
    return tf.concat([preprocess_notes(x), preprocess_octaves(x)], axis=1)


def preprocess_durations(x):
    x = tf.cast(x, tf.int32)
    duration = x[:, 2]
    oh_duration = layers.experimental.preprocessing.CategoryEncoding(num_tokens=7, output_mode="one_hot")(duration)
    return oh_duration


def preprocess_dotted(x):
    dotted = x[:, 3]
    dotted = tf.reshape(tf.cast(dotted, tf.float32),(-1,1))
    return dotted

def duration_and_dotted(x):
    return tf.concat([preprocess_durations(x), preprocess_dotted(x)], axis=1)


def preprocess_labels(x,number_of_classes):
    options = {
        "notes": preprocess_notes,
        "octaves": preprocess_octaves,
        "duration": preprocess_durations,
        "dotted": preprocess_dotted,
    }
    processed = {}
    for key,value in number_of_classes.items():
        if value:
            processed[key]=options[key](x)
    return processed

def preprocess_data(data: tf.data.Dataset, label_features: dict):
    window_size = data.element_spec[0].shape[0]
    n_features = data.element_spec[0].shape[1]
    inputs = keras.Input(shape=(window_size, n_features), dtype=data.element_spec[0].dtype)
    preprocessed_inputs = preprocess_inputs(inputs)
    raw_labels = keras.Input(shape=(n_features), dtype=data.element_spec[0].dtype)
    preprocessed_labels = preprocess_labels(raw_labels, label_features)
    in_prep_model = tf.keras.Model(inputs=inputs, outputs=preprocessed_inputs)
    out_prep_model = tf.keras.Model(inputs=raw_labels, outputs=preprocessed_labels)
    plot_model(in_prep_model, to_file='input.png')
    plot_model(out_prep_model, to_file='label.png')

    return preprocessed_inputs, in_prep_model, out_prep_model
