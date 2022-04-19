import os
import tensorflow as tf
import numpy as np
from tensorflow.python.keras.utils.vis_utils import plot_model

from preprocessing import preprocessing


class Dataset:
    def __init__(self, input_beats: int, batch_size: int, output_path: str, in_prep_name: str, out_prep_name: str):
        self.window_path = os.path.join("data", "modest")
        self.output_path = output_path
        self.window_size = input_beats + 1
        self.input_beats = input_beats
        self.batch_size = batch_size
        self.train = self.extract_and_preprocess("train", in_prep_name, out_prep_name)
        self.input_shape = (self.input_beats, 4)
        self.test = self.extract_and_preprocess("test", in_prep_name, out_prep_name)
        folder_path = os.path.join(self.window_path, "train", str(self.window_size))
        self.class_weights = {
            "semitone": np.load(os.path.join(folder_path, "semitone_weights.npy")),
            "octave": np.load(os.path.join(folder_path, "octave_weights.npy")),
            "dur_log": np.load(os.path.join(folder_path, "dur_log_weights.npy")),
            "dotted": np.load(os.path.join(folder_path, "dotted_weights.npy"))
        }
        self.number_of_classes = {feature: self.class_weights[feature].shape[-1] for feature in self.class_weights}

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
        # plot_model(
        #     in_prep_model,
        #     to_file=os.path.join(
        #        self.output_path,
        #        'in_prep_model.png'
        #     ),
        #     show_shapes=True, show_layer_names=False
        # )
        # plot_model(
        #     out_prep_model,
        #     to_file=os.path.join(
        #         self.output_path,
        #         'out_prep_model.png'
        #     ),
        #     show_shapes=True, show_layer_names=False
        # )
        ds = ds.shuffle(buffer_size=1000)
        ds = ds.batch(self.batch_size, drop_remainder=True)
        ds = ds.map(
            lambda x, y: (in_prep_model(x), out_prep_model(y)),
            num_parallel_calls=tf.data.AUTOTUNE
        ).prefetch(tf.data.AUTOTUNE)
        return ds
