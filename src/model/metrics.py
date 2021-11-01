import numpy as np
import tensorflow as tf

from sklearn.metrics import confusion_matrix


def balanced_accuracy(y_true, y_pred):
    y_pred_index = tf.argmax(y_pred, axis=-1)
    y_pred_oh = tf.one_hot(y_pred_index, tf.shape(y_true)[1])
    samples_per_class = tf.reduce_sum(y_true, axis=0)
    hits_per_class = tf.reduce_sum(y_true * y_pred_oh, axis=0)
    accuracy_per_class = hits_per_class / (samples_per_class + 1e-8)
    n_represented_classes = tf.reduce_sum(tf.sign(samples_per_class))
    return tf.reduce_sum(accuracy_per_class) / n_represented_classes


class BalancedAccuracy(tf.keras.metrics.Metric):

    def __init__(self, shape: int, name='balanced_accuracy', **kwargs):
        super(BalancedAccuracy, self).__init__(name=name, **kwargs)
        self.shape = shape
        self.counts = tf.zeros([shape])
        self.hits = tf.zeros([shape])

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred_index = tf.argmax(y_pred, axis=-1)
        y_pred_oh = tf.one_hot(y_pred_index, tf.shape(y_true)[-1])
        self.hits += tf.reduce_sum(y_true * y_pred_oh, axis=0)
        self.counts += tf.reduce_sum(y_true, axis=0)


    def result(self):
        recalls=tf.math.divide_no_nan(
            self.hits,
            self.counts
        )
        # compute mean discarding labels with no counts for this epoch
        return tf.reduce_sum(recalls)/tf.reduce_sum(tf.sign(self.counts))

    def reset_states(self):
        self.counts = tf.zeros(self.shape)
        self.hits = tf.zeros(self.shape)


def get_metric(name, shape: int):
    metrics = {
        # "ba": BalancedAccuracy(shape)
        'ba': balanced_accuracy,
        'ac': "accuracy"
    }
    return metrics[name]