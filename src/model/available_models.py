import tensorflow.keras as keras
from tensorflow.keras import layers


def lstm_model():
    inputs = layers.Input((5, 13))
    x = layers.LSTM(65, activation='relu', input_shape=(5, 13))(inputs)
    output1 = layers.Dense(13, activation='softmax', name='notes')(x)
    output2 = layers.Dense(8, activation='softmax', name='duration')(x)
    return keras.Model(inputs=inputs, outputs=[output1, output2])


def ffwd_model():
    inputs = layers.Input((5, 13))
    x = layers.Flatten()(inputs)
    x = layers.Dense(40, activation='relu')(x)
    x = layers.Dense(30, activation='relu')(x)
    x = layers.Dense(20, activation='relu')(x)
    output1 = layers.Dense(13, activation='softmax', name='notes')(x)  # cross entropy
    output2 = layers.Dense(8, activation='softmax', name='duration')(x)
    return keras.Model(inputs=inputs, outputs=[output1, output2])


available_models = {
    "ffwd": ffwd_model,
    "lstm": lstm_model
}
