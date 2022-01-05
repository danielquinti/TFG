import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K


def weighted_cce(weights):
    def loss(y_true, y_pred):
        y_pred_clip = K.clip(y_pred, K.epsilon(), 1)
        return K.sum(-weights * y_true * K.log(y_pred_clip), axis=-1)

    return loss


def get_loss_function(name, weights):
    losses = {
        "wcce": weighted_cce(weights),
        "cce": tf.keras.losses.CategoricalCrossentropy(),
        "1cce": weighted_cce(np.ones_like(weights))
    }
    return losses[name]
