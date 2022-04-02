import os
import tensorflow as tf
import numpy as np




class Dataset:
    def __init__(self, input_beats: int):
        window_path = os.path.join("..", "data", "windowed")
        window_size = input_beats+1
        train = np.load(os.path.join(window_path, "train", f'{window_size}.npy')).reshape(-1, window_size, 4)
        self.train = tf.data.Dataset.from_tensor_slices(
            (
                train[:, :input_beats, :],
                train[:, input_beats, :]
            )
        )
        test = np.load(os.path.join(window_path, "test", f'{window_size}.npy')).reshape(-1, window_size, 4)
        self.test = tf.data.Dataset.from_tensor_slices(
            (
                test[:, :input_beats, :],
                test[:, input_beats, :]
            )
        )
        self.number_of_classes = {
            "notes": 121,
            "duration": 14,
        }
