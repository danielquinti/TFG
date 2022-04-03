import os
import tensorflow as tf
import numpy as np

from architecture import preprocessing

def expand_label_features(active_features):
    result= {}
    if active_features["notes"]:
        result["notes"] = 13
        result["octaves"] = 10
    if active_features["duration"]:
        result["duration"] = 7
        result["dotted"] = 1
    return result
class Dataset:
    def __init__(self, input_beats: int, batch_size, active_features: dict):
        window_path = os.path.join("..", "data", "windowed")
        self.window_size = input_beats+1
        self.input_beats = input_beats
        self.batch_size = batch_size
        self.number_of_classes = expand_label_features(active_features)
        self.train = self.extract_dataset("train", window_path)
        self.test = self.extract_dataset("test", window_path)
        self.weights = {
            "notes": np.load(os.path.join(window_path, "train", str(self.window_size), "nw.npy")),
            "duration": np.load(os.path.join(window_path, "train", str(self.window_size), "dw.npy"))
        }

    def extract_dataset(self, dist_name, window_path):
        folder_path = os.path.join(window_path, dist_name, str(self.window_size))
        data_path = os.path.join(folder_path, "windows.npy")
        data = np.load(data_path).reshape(-1, self.window_size, 4)
        ds = tf.data.Dataset.from_tensor_slices(
            (
                data[:, :self.input_beats, :],
                data[:, self.input_beats, :]
            )
        )
        preprocessed_inputs, in_prep_model, out_prep_model = preprocessing.preprocess_data(ds, self.number_of_classes)
        self.input_shape = preprocessed_inputs.shape[1:]
        ds = ds.shuffle(buffer_size=self.batch_size * 2)
        ds = ds.batch(self.batch_size)
        ds = ds.map(
            lambda x, y: (in_prep_model(x), out_prep_model(y)),
            num_parallel_calls=tf.data.AUTOTUNE
        ).prefetch(tf.data.AUTOTUNE)
        return ds