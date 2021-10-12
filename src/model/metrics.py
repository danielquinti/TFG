import numpy as np
import tensorflow as tf


def mean_ap(y_true, y_pred):
    obtained = tf.math.argmax(y_pred)
    expected = tf.math.argmax(y_true)
    cm = tf.math.confusion_matrix(expected, obtained)
    return tf.math.reduce_mean(tf.math.divide(
        tf.linalg.diag(cm),
        tf.math.reduce_sum(cm, axis=1)
    )
    )
