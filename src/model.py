#!/usr/bin/python3
import itertools

import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras as keras
from sklearn.metrics import confusion_matrix
from tensorflow.keras import layers
import time
from DatasetFromCSV import *


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


def compile_and_train_lstm(dataset):
    inputs = layers.Input((5, 13))
    x = layers.LSTM(65, activation='relu', input_shape=(5, 13))(inputs)
    output1 = layers.Dense(13, activation='softmax', name='notes')(x)  # cross entropy
    output2 = layers.Dense(1, activation='sigmoid', name='duration')(x)  # mse
    model = keras.Model(inputs=inputs, outputs=[output1, output2])
    model.compile(
        loss=dict(notes=tf.keras.losses.CategoricalCrossentropy(), duration=tf.keras.losses.MeanSquaredError()),
        optimizer='adam',
        metrics=dict(notes=tf.keras.metrics.Accuracy(), duration=keras.metrics.MeanAbsoluteError()),
        loss_weights=dict(notes=1, duration=0.01)
    )

    model.fit(x=dataset.train_inputs,
              y={'notes': dataset.train_labels["notes"], 'duration': dataset.train_labels["duration"]},
              epochs=MAX_EPOCHS,
              batch_size=BATCH_SIZE,
              verbose=2,
              shuffle=True,
              validation_split=0.2
              )
    return model

def compile_and_train_ffwd(dataset):
    inputs = layers.Input((5, 13))
    x = layers.Flatten()(inputs)
    x = layers.Dense(40, activation='relu')(x)
    x = layers.Dense(30, activation='relu')(x)
    x = layers.Dense(20, activation='relu')(x)
    output1 = layers.Dense(13, activation='softmax', name='notes')(x)  # cross entropy
    output2 = layers.Dense(1, activation='sigmoid', name='duration')(x)  # mse
    model = keras.Model(inputs=inputs, outputs=[output1, output2])
    model.compile(
        loss=dict(notes=tf.keras.losses.CategoricalCrossentropy(), duration=tf.keras.losses.MeanSquaredError()),
        optimizer='adam',
        metrics=dict(notes="categorical_accuracy", duration=keras.metrics.MeanAbsoluteError()),
        loss_weights=dict(notes=1, duration=0.01)
    )

    model.fit(x=dataset.train_inputs,
              y={'notes': dataset.train_labels["notes"], 'duration': dataset.train_labels["duration"]},
              epochs=MAX_EPOCHS,
              batch_size=BATCH_SIZE,
              verbose=2,
              shuffle=True,
              validation_split=0.2
              )
    return model


def build_and_plot_cm(model,name):

    predictions = model.predict(dataset.test_inputs, verbose=0)

    # notes confusion matrix
    obtained = np.argmax(predictions[0], axis=1).flatten()
    expected = np.argmax(dataset.test_labels["notes"], axis=1).flatten()
    matrix = confusion_matrix(y_true=expected, y_pred=obtained)
    print(matrix)
    np.savetxt(("stats\\"+name+"_notes.csv"), matrix, fmt='%i')
    # cm_plot_labels = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B", "REST"]
    # plot_confusion_matrix(matrix, cm_plot_labels)

    # duration confusion matrix
    obtained = np.round(np.log2(predictions[1]))
    expected = np.round(np.log2(dataset.test_labels["duration"])).flatten()
    matrix = confusion_matrix(y_true=expected, y_pred=obtained)
    print(matrix)
    np.savetxt(("stats\\"+name+"_duration.csv"), matrix, fmt='%i')
    # cm_plot_labels = ["1", "1/2", "1/4", "1/8", "1/16", "1/32", "1/64", "1/128"]
    # plot_confusion_matrix(matrix, cm_plot_labels)

BATCH_SIZE = 32
MAX_EPOCHS = 300
dummy = False
if __name__ == "__main__":
    start=time.time()
    dataset = DatasetFromCSV(5, 1, dummy)

    print("---------------------LSTM TRAINING-----------------")
    lstm = compile_and_train_lstm(dataset)
    print("---------------------LSTM CM-----------------")
    build_and_plot_cm(lstm,"lstm")

    print("---------------------BASELINE TRAINING-----------------")
    baseline = compile_and_train_ffwd(dataset)
    print("---------------------BASELINE CM-----------------")
    build_and_plot_cm(baseline,"baseline")
    print("--- %s seconds ---" % (time.time() - start))
