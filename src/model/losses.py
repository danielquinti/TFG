import numpy as np
import tensorflow as tf


def weighted_cce_n(y_true, y_pred):
    weights = [0.85787272, 1.49633911, 0.61418581, 1.65545341, 0.48246767,
               1.13726069, 0.92723318, 0.6441895, 1.29881069, 0.54618119,
               1.76055792, 0.64324125, 0.93620685]
    return tf.keras.losses.CategoricalCrossentropy()(
        tf.math.multiply(
            y_true,
            weights),
        tf.math.multiply(
            y_pred,
            weights)
    )


def weighted_cce_d(y_true, y_pred):
    weights = np.array([1.27864199e-01, 4.72674192e-02, 8.78259833e-03, 1.35276014e-03,
                        2.25467030e-03, 2.49283201e-02, 4.71568068e-01, 7.31598197e+00])
    return tf.keras.losses.CategoricalCrossentropy()(
        tf.math.multiply(
            y_true,
            weights),
        tf.math.multiply(
            y_pred,
            weights)
    )
