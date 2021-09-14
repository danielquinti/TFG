#!/usr/bin/python3

import os
import numpy as np
import IPython
import IPython.display
import pandas as pd
from src.WindowGenerator import *
import tensorflow as tf
from tensorflow.keras import layers
from sklearn.metrics import confusion_matrix
import itertools
import matplotlib.pyplot as plt

N = 6
input_size = 12
output_size = 12
seq_length = N - 1
batch_size = 32
MAX_EPOCHS =30

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

def compile_and_fit(model, window, patience=4):
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                      patience=patience,
                                                      mode='min')

    model.compile(loss=tf.losses.MeanSquaredError(),
                  optimizer=tf.optimizers.Adam(),
                  metrics=[tf.metrics.MeanAbsoluteError()])

    history = model.fit(window.train, epochs=MAX_EPOCHS,
                        validation_data=window.val,
                        batch_size=batch_size,
                        callbacks=[early_stopping],shuffle=False)
    return history


os.chdir("data")
train = pd.read_csv("train.csv")
val = pd.read_csv("dev.csv")
test = pd.read_csv("test.csv")
conv_window = WindowGenerator(5, 1, 1, train, val, test)
prueba = test.to_numpy()
prueba_x = prueba[np.mod(np.arange(np.shape(prueba)[0]), 6) != 0]
prueba_y = prueba[np.mod(np.arange(np.shape(prueba)[0]), 6) == 0]
prueba_x = prueba_x.reshape(np.shape(prueba_y)[0], 5, 12)
multi_step_dense = tf.keras.Sequential([
    # layers.Bidirectional(layers.LSTM(units=128,input_shape=(5, 12))),
    # layers.Dropout(rate=0.2),
    # layers.Dense(units=12)

    layers.Flatten(),
    tf.keras.layers.Dense(units=12, activation='relu'),
    tf.keras.layers.Dense(units=12, activation='relu'),
    tf.keras.layers.Dense(units=12),
    tf.keras.layers.Reshape([1, -1])
])

compile_and_fit(multi_step_dense, conv_window)
# print(multi_step_dense.predict(train,verbose=0))
predicciones = multi_step_dense.predict(prueba_x, verbose=0)


predicciones=np.argmax(predicciones,axis=2).flatten()
prueba_y=np.argmax(prueba_y,axis=1)
cm = confusion_matrix(y_true=prueba_y, y_pred=predicciones)
cm_plot_labels=["0","1","2","3","4","5","6","7","8","9","10","11"]
plot_confusion_matrix(cm,cm_plot_labels)
