from abc import ABC, abstractmethod
import tensorflow as tf
from tensorflow.keras import layers


class BeatPreprocessor(ABC):
    @abstractmethod
    def process(self, x):
        pass


class SemitoneExtractor(BeatPreprocessor):
    def process(self, x):
        x = tf.cast(x, tf.int32)
        return tf.reshape(x[:, 0], [-1, 1])


class OctaveExtractor(BeatPreprocessor):
    def process(self, x):
        x = tf.cast(x, tf.int32)
        return tf.reshape(x[:, 1], [-1, 1])


class DurLogExtractor(BeatPreprocessor):
    def process(self, x):
        x = tf.cast(x, tf.int32)
        return tf.reshape(x[:, 2], [-1, 1])


class DottedExtractor(BeatPreprocessor):
    def process(self, x):
        x = tf.cast(x, tf.int32)
        return tf.reshape(x[:, 3], [-1, 1])


class SemitoneOHPreprocessor(BeatPreprocessor):
    def process(self, x):
        x = tf.cast(x, tf.int32)
        semitone = x[:, 0]
        return tf.one_hot(semitone, 13)


class OctaveOHPreprocessor(BeatPreprocessor):
    def process(self, x):
        x = tf.cast(x, tf.int32)
        octave = x[:, 1]
        return tf.one_hot(octave, 10)


class DurLogOHPreprocessor(BeatPreprocessor):
    def process(self, x):
        x = tf.cast(x, tf.int32)
        dur_log = x[:, 2]
        return tf.one_hot(dur_log, 7)


class DottedOHPreprocessor(BeatPreprocessor):
    def process(self, x):
        x = tf.cast(x, tf.int32)
        dotted = x[:, 3]
        return tf.one_hot(dotted, 2)


class AllOHPreprocessor(BeatPreprocessor):
    def process(self, x):
        return tf.concat(
            [
                SemitoneOHPreprocessor().process(x),
                OctaveOHPreprocessor().process(x),
                DurLogOHPreprocessor().process(x),
                DottedOHPreprocessor().process(x)
            ],
            axis=1
        )


class AllSingleValuePreprocessor(BeatPreprocessor):
    def process(self, x):
        return \
            10 * SemitoneExtractor().process(x) + \
            7 * OctaveExtractor().process(x) + \
            2 * DurLogExtractor().process(x) + \
            DottedExtractor().process(x)


class PitchSingleValuePreprocessor(BeatPreprocessor):
    def process(self, x):
        return \
            12 * OctaveExtractor().process(x) + \
            SemitoneExtractor().process(x)


class DurationSingleValuePreprocessor(BeatPreprocessor):
    def process(self, x):
        return \
            2 * DurLogExtractor().process(x) + \
            DottedExtractor().process(x)


class AllOHSingleValuePreprocessor(BeatPreprocessor):
    def process(self, x):
        return tf.concat(
            [
                layers.experimental.preprocessing.CategoryEncoding(num_tokens=133, output_mode="one_hot")
                (PitchSingleValuePreprocessor().process(x)),
                layers.experimental.preprocessing.CategoryEncoding(num_tokens=13, output_mode="one_hot")
                (DurationSingleValuePreprocessor().process(x)),
            ], axis=1
        )


class DurationOHConcatPreprocessor(BeatPreprocessor):
    def process(self, x):
        return tf.concat(
            [
                DurLogOHPreprocessor().process(x),
                DottedOHPreprocessor().process(x)
            ],
            axis=1
        )


class PitchOHConcatPreprocessor(BeatPreprocessor):
    def process(self, x):
        return tf.concat(
            [
                SemitoneOHPreprocessor().process(x),
                OctaveOHPreprocessor().process(x())
            ],
            axis=1
        )


class AllSeparatePreprocessor(BeatPreprocessor):
    def process(self, x):
        return {
            "semitone": SemitoneOHPreprocessor().process(x),
            "octave": OctaveOHPreprocessor().process(x),
            "dur_log": DurLogOHPreprocessor().process(x),
            "dotted": DottedOHPreprocessor().process(x)
        }


class NoPreprocessor(BeatPreprocessor):
    def process(self, x):
        return x
