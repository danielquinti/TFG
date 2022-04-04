import tensorflow as tf
import inspect


def get_optimizer(data: dict):
    name = data["name"]
    options = {
        "adam": tf.keras.optimizers.Adam,
        "rmsprop": tf.keras.optimizers.RMSprop,
        "sgd": tf.keras.optimizers.SGD
    }
    signature = inspect.getfullargspec(options[name])

    params = dict(zip(signature.args[1:], signature.defaults))
    for k, v in list(data.items())[1:]:
        if params.get(k):
            params[k] = v
        else:
            raise ValueError("Malformed optimizer configuration")
    return options[name](*(params.values()))
