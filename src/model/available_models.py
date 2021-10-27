import tensorflow.keras as keras
from tensorflow.keras import layers

from src.model.losses import get_loss_function
from src.model.metrics import get_metric


def lstm_model(n_classes, d_classes, input_beats, label_beats):
    inputs = layers.Input((input_beats, n_classes))
    x = layers.LSTM(65, activation='relu', input_shape=(input_beats, n_classes))(inputs)
    output1 = layers.Dense(n_classes, activation='softmax', name='notes')(x)
    output2 = layers.Dense(d_classes, activation='softmax', name='duration')(x)
    return keras.Model(inputs=inputs, outputs=[output1, output2])


def ffwd_model(n_classes, d_classes, input_beats, label_beats):
    inputs = layers.Input((input_beats, n_classes))
    x = layers.Flatten()(inputs)
    x = layers.Dense(40, activation='relu')(x)
    x = layers.Dense(30, activation='relu')(x)
    x = layers.Dense(20, activation='relu')(x)
    output1 = layers.Dense(n_classes, activation='softmax', name='notes')(x)  # cross entropy
    output2 = layers.Dense(d_classes, activation='softmax', name='duration')(x)
    return keras.Model(inputs=inputs, outputs=[output1, output2])


available_models = {
    "ffwd": ffwd_model,
    "lstm": lstm_model
}