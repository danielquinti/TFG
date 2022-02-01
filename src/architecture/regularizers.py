from tensorflow.keras import regularizers


def get_regularizer(data):
    return regularizers.l2(data)
