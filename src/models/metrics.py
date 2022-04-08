import tensorflow as tf

#TODO name precision
class BalancedAccuracy(tf.keras.metrics.Metric):

    def __init__(self, shape: int, name='balanced_accuracy', **kwargs):
        super(BalancedAccuracy, self).__init__(name=name, **kwargs)
        self.shape = shape
        self.counts = tf.Variable(tf.zeros([shape]))
        self.hits = tf.Variable(tf.zeros([shape]))

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred_index = tf.argmax(y_pred, axis=-1)
        y_pred_oh = tf.one_hot(y_pred_index, tf.shape(y_true)[-1])
        self.hits.assign_add(tf.reduce_sum(y_true * y_pred_oh, axis=0))
        self.counts.assign_add(tf.reduce_sum(y_true, axis=0))

    def result(self):
        recalls = tf.math.divide_no_nan(
            self.hits,
            self.counts
        )
        # compute mean discarding labels with no counts
        return tf.reduce_sum(recalls) / tf.reduce_sum(tf.sign(self.counts))

    def reset_states(self):
        self.counts.assign(tf.zeros([self.shape]))
        self.hits.assign(tf.zeros([self.shape]))


class GeometricAccuracy(tf.keras.metrics.Metric):

    def __init__(self, shape: int, name='geometric_accuracy', **kwargs):
        super(GeometricAccuracy, self).__init__(name=name, **kwargs)
        self.shape = shape
        self.counts = tf.Variable(tf.zeros([shape]))
        self.hits = tf.Variable(tf.zeros([shape]))

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred_index = tf.argmax(y_pred, axis=-1)
        y_pred_oh = tf.one_hot(y_pred_index, tf.shape(y_true)[-1])
        self.hits.assign_add(tf.reduce_sum(y_true * y_pred_oh, axis=0))
        self.counts.assign_add(tf.reduce_sum(y_true, axis=0))

    def result(self):
        recalls = tf.math.divide_no_nan(
            self.hits,
            self.counts
        )
        # compute geometric mean discarding labels with no counts

        tf.print(tf.where(recalls > 0))
        nonzeroprod = tf.reduce_prod(
            recalls + 1e-8
        )
        exponent = tf.math.divide(
            1.0,
            tf.reduce_sum(
                tf.sign(self.counts)
            )
        )
        return tf.math.pow(
            nonzeroprod,
            exponent
        )

    def reset_states(self):
        self.counts.assign(tf.zeros([self.shape]))
        self.hits.assign(tf.zeros([self.shape]))


def get_metric(name, shape: int):
    metrics = {
        "ba": BalancedAccuracy(shape),
        "ga": GeometricAccuracy(shape),
        'ac': "accuracy",
        're': tf.keras.metrics.Recall(),
    }
    return metrics[name]
