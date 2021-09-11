#!/usr/bin/python3

import os
import numpy as np
import IPython
import IPython.display
import pandas as pd
from src.WindowGenerator import *
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


N=6
input_size=12
output_size=12
seq_length=N-1
batch_size=64
MAX_EPOCHS = 20

def compile_and_fit(model, window, patience=2):
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                    patience=patience,
                                                    mode='min')

    model.compile(loss=tf.losses.MeanSquaredError(),
                optimizer=tf.optimizers.Adam(),
                metrics=[tf.metrics.MeanAbsoluteError()])

    history = model.fit(window.train, epochs=MAX_EPOCHS,
                      validation_data=window.val,
                      callbacks=[early_stopping])
    return history

os.chdir("data")
train=pd.read_csv("train.csv")
val=pd.read_csv("dev.csv")
test=pd.read_csv("test.csv")

conv_window = WindowGenerator(5, 1, 1,train,val,test)
multi_step_dense = tf.keras.Sequential([
    # Shape: (time, features) => (time*features)
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(units=12, activation='relu'),
    tf.keras.layers.Dense(units=12, activation='relu'),
    tf.keras.layers.Dense(units=12),
    # Add back the time dimension.
    # Shape: (outputs) => (1, outputs)
    tf.keras.layers.Reshape([1, -1]),
])
history = compile_and_fit(multi_step_dense, conv_window)
# IPython.display.clear_output()
# val_performance = dict()
# performance = dict()
# val_performance['Multi step dense'] = multi_step_dense.evaluate(conv_window.val)
# performance['Multi step dense'] = multi_step_dense.evaluate(conv_window.test, verbose=0)
# print(val_performance)
# print(performance)