import tensorflow.keras.backend as K


def weighted_cce(weights):
    def loss(y_true, y_pred):
        y_pred_clip = K.clip(y_pred, K.epsilon(), 1)
        return K.sum(-weights * y_true * K.log(y_pred_clip), axis=-1)

    return loss
