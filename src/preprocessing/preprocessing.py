import tensorflow as tf
from tensorflow.keras.utils import plot_model
from preprocessing import beat_preprocessors as bp


# decorator pattern
class WindowPreprocessor:
    def __init__(self, prep: bp.BeatPreprocessor):
        self.prep = prep

    def process(self, x):
        if type(self.prep) is bp.NoPreprocessor:
            return x
        window_size = x.shape[1]
        x = tf.reshape(x, (-1, 4))
        x = self.prep.process(x)
        if type(x) == dict:
            return {name: tf.reshape(data, (-1, window_size)) for name, data in x.items()}
        return tf.reshape(x, shape=[-1, window_size, x.shape[-1]])


class DataPreprocessor:
    def __init__(self, data: tf.data.Dataset, in_criteria: str, out_criteria: str):
        processors = {
            "se": bp.SemitoneExtractor,
            "oe": bp.OctaveExtractor,
            "dle": bp.DurLogExtractor,
            "dte": bp.DottedExtractor,

            "soh": bp.SemitoneOHPreprocessor,
            "ooh": bp.OctaveOHPreprocessor,
            "dloh": bp.DurLogOHPreprocessor,
            "dtoh": bp.DottedOHPreprocessor,

            "aoh": bp.AllOHPreprocessor,

            "asv": bp.AllSingleValuePreprocessor,
            "psv": bp.PitchSingleValuePreprocessor,
            "dsv": bp.DurationSingleValuePreprocessor,

            "aohsv": bp.AllOHSingleValuePreprocessor,
            "dohc": bp.DurationOHConcatPreprocessor,
            "pohc": bp.PitchOHConcatPreprocessor,

            "as": bp.AllSeparatePreprocessor,
            "np": bp.NoPreprocessor
        }
        self.window_size = data.element_spec[0].shape[0]
        self.n_features = data.element_spec[0].shape[1]
        self.data = data

        self.in_prep = WindowPreprocessor(processors[in_criteria]())
        self.out_prep: bp.BeatPreprocessor = processors[out_criteria]()

    def preprocess(self):
        inputs = tf.keras.Input(shape=(self.window_size, self.n_features), dtype=self.data.element_spec[0].dtype)
        preprocessed_inputs = self.in_prep.process(inputs)
        raw_labels = tf.keras.Input(shape=self.n_features, dtype=self.data.element_spec[0].dtype)
        preprocessed_labels = self.out_prep.process(raw_labels)
        in_prep_model = tf.keras.Model(inputs=inputs, outputs=preprocessed_inputs)
        out_prep_model = tf.keras.Model(inputs=raw_labels, outputs=preprocessed_labels)
        return in_prep_model, out_prep_model
