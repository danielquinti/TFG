#!/usr/bin/python3
import itertools

import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras as keras
from sklearn.metrics import confusion_matrix
from tensorflow.keras import layers

from DatasetFromCSV import *

N = 6
input_size = 13
output_size = 13
seq_length = N - 1
batch_size = 32
MAX_EPOCHS = 5


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


def compile_and_train(dataset):
    inputs = layers.Input((5, 13))
    x = layers.LSTM(65, activation='relu', input_shape=(5, 13))(inputs)
    output1 = layers.Dense(13, activation='softmax', name='notes')(x)  # cross entropy
    output2 = layers.Dense(1, activation='relu', name='duration')(x)  # mse
    model = keras.Model(inputs=inputs, outputs=[output1, output2])
    model.compile(
        loss=dict(notes=tf.keras.losses.CategoricalCrossentropy(), duration=tf.keras.losses.MeanSquaredError()),
        optimizer='adam',
        metrics=dict(notes=keras.metrics.MeanAbsoluteError(), duration=keras.metrics.MeanAbsoluteError()))

    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                      patience=2,
                                                      mode='min')

    return model.fit(x=dataset.train_inputs,
                     y={'notes': dataset.train_labels["notes"], 'duration': dataset.train_labels["duration"]},
                     epochs=MAX_EPOCHS,
                     batch_size=batch_size,
                     verbose=2,
                     shuffle=False,
                     # callbacks=[early_stopping],
                     # validation_split=0.2
                     loss_weights=dict(notes=1, duration=0.01)
                     )


def analyze_results(model):
    predictions = model.predict(dataset.test_inputs, verbose=0)
    obtained = np.argmax(predictions[0], axis=1).flatten()
    expected = np.argmax(dataset.test_labels["notes"], axis=2).flatten()
    matrix = confusion_matrix(y_true=expected, y_pred=obtained)
    cm_plot_labels = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11,12"]
    plot_confusion_matrix(matrix, cm_plot_labels)

    predictions = model.predict(dataset.test_inputs, verbose=0)
    obtained = np.round(np.log2(predictions[1]))
    expected = np.round(np.log2(dataset.test_labels["duration"])).flatten()
    matrix = confusion_matrix(y_true=expected, y_pred=obtained)
    cm_plot_labels = ["whole", "half", "quarter", "eighth", "sixteenth", "thirty-second", "sixty-fourth",
                      "hundred-twenty-eighth"]
    plot_confusion_matrix(matrix, cm_plot_labels)


if __name__ == "__main__":
    dataset = DatasetFromCSV(5, 1)
    notes=dataset.test_inputs
    notes = np.argmax(notes,axis=2).flatten()
    plt.hist(notes, bins=13)
    plt.show()
