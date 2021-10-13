import tensorflow as tf


def balanced_accuracy(y_true, y_pred):
    y_pred_index = tf.argmax(y_pred, axis=-1)
    y_pred_oh = tf.one_hot(y_pred_index, tf.shape(y_true)[1])
    samples_per_class = tf.reduce_sum(y_true, axis=0)
    hits_per_class = tf.reduce_sum(y_true * y_pred_oh, axis=0)
    accuracy_per_class = hits_per_class / (samples_per_class + 1e-8)
    n_represented_classes = tf.reduce_sum(tf.sign(samples_per_class))
    return tf.reduce_sum(accuracy_per_class) / n_represented_classes
