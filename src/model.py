#!/usr/bin/python3
import math
import os
import numpy as np
import IPython
import time
import IPython.display
import pandas as pd
from WindowGenerator import *
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers
from sklearn.metrics import confusion_matrix
import itertools
import matplotlib as mpl
import matplotlib.pyplot as plt

N = 6
input_size = 13
output_size = 13
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

# start=time.time()
# print(time.time()-start)

# dataset = WindowGenerator(5, 1,0.2)
# os.chdir("data")
# np.savetxt("train_inputs.csv", dataset.train_inputs, fmt='%1.6f')
# np.savetxt("train_labels.csv", dataset.train_labels, fmt='%1.6f')
# np.savetxt("test_inputs.csv", dataset.test_inputs, fmt='%1.6f')
# np.savetxt("test_labels.csv", dataset.test_labels, fmt='%1.6f')
dataset=WindowGenerator(5,1,0.2)

# create and fit the LSTM network
# model = keras.Sequential()
# model.add(layers.LSTM(4, input_shape=(5, 13)))
# model.add(layers.Dense(1))
# model.compile(loss='mean_squared_error', optimizer='adam')
# model.fit(dataset.train_inputs, dataset.train_labels["notes"], epochs=100, batch_size=1, verbose=2)

inputs = layers.Input((5,13))
x = layers.Dense(13, activation='relu')(inputs)
x = layers.Dense(13, activation='relu')(x)
output1 = layers.Dense(13,activation='softmax',name='notes')(x)# cross entropy
output2 = layers.Dense(1, activation='relu',name='duration')(x)#mse
model = keras.Model(inputs=inputs, outputs=[output1, output2])
model.compile(loss='mse',
              optimizer='adam',
              metrics={"notes":keras.metrics.MeanAbsoluteError(),
              'duration': keras.metrics.MeanAbsoluteError()})

early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                  patience=4,
                                                  mode='min')

history = model.fit(x=dataset.train_inputs,
                    y={'notes': dataset.train_labels["notes"]
                        , 'duration':dataset.train_labels["duration"]
                    },
                    epochs=MAX_EPOCHS, batch_size=batch_size,verbose=2,shuffle=False,callbacks=[early_stopping],validation_split=0.2)
#TODO las predicciones tienen forma 5*13
print("done")
predicciones=model.predict(dataset.test_inputs,verbose=0)
obtained=np.argmax(predicciones[0],axis=1).flatten()
expected=np.argmax(dataset.test_labels["notes"],axis=2).flatten()
cm = confusion_matrix(y_true=expected, y_pred=obtained)
cm_plot_labels=["0","1","2","3","4","5","6","7","8","9","10","11,12"]
plot_confusion_matrix(cm,cm_plot_labels)