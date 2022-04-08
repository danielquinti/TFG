import os
import tensorflow as tf
import numpy as np
from tensorflow.python.keras.utils.vis_utils import plot_model

from src.preprocessing import preprocessing


class Dataset:
    def __init__(self, input_beats: int, batch_size: int, in_prep_name: str, out_prep_name: str):
        self.window_path = os.path.join("data", "windowed")
        self.window_size = input_beats + 1
        self.input_beats = input_beats
        self.batch_size = batch_size
        self.train = self.extract_and_preprocess("train", in_prep_name, out_prep_name)
        self.input_shape = (self.input_beats, 4)
        self.test = self.extract_and_preprocess("test", in_prep_name, out_prep_name)
        self.number_of_classes = {
            "semitone": 13,
            "octave": 10,
            "dur_log": 7,
            "dotted": 2
        }

    def extract_and_preprocess(self, dist_name: str, in_prep_name: str, out_prep_name: str):
        folder_path = os.path.join(self.window_path, dist_name, str(self.window_size))
        data_path = os.path.join(folder_path, "windows.npy")
        data = np.load(data_path).reshape((-1, self.window_size, 4))
        ds = tf.data.Dataset.from_tensor_slices(
            (
                data[:, :self.input_beats, :],
                data[:, self.input_beats, :]
            )
        )
        in_prep_model, out_prep_model = preprocessing.DataPreprocessor(
            ds,
            in_prep_name,
            out_prep_name
        ).preprocess()
        plot_model(in_prep_model, to_file="in_prep_model.png", show_shapes=True)
        plot_model(out_prep_model, to_file="out_prep_model.png", show_shapes=True)
        ds = ds.shuffle(buffer_size=self.batch_size * 2)
        ds = ds.batch(self.batch_size, drop_remainder=True)
        ds = ds.map(
            lambda x, y: (in_prep_model(x), out_prep_model(y)),
            num_parallel_calls=tf.data.AUTOTUNE
        ).prefetch(tf.data.AUTOTUNE)
        return ds
