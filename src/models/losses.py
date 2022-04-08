import inspect

import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K


def weighted_cce(weights):
    def loss(y_true, y_pred):
        y_pred_clip = K.clip(y_pred, K.epsilon(), 1)
        return K.sum(-weights * y_true * K.log(y_pred_clip), axis=-1)

    return loss


def get_loss_function(name):
    losses = {
        "wcce": tf.keras.losses.CategoricalCrossentropy(),
        "cce": tf.keras.losses.CategoricalCrossentropy(),
    }
    return losses[name]
