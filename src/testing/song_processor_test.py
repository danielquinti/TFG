import os
import unittest
import numpy as np
import sys

sys.path.append(
    os.path.join(
        "src",
        "data_processing"
    )
)
from src.data_processing import song_processor


class SongProcessorTest(unittest.TestCase):
    def setUp(self):
        self.input_path = os.path.join(
            os.getcwd(),
            "src",
            "testing",
            "test_input"
        )
        self.output_path = os.path.join(
            os.getcwd(),
            "src",
            "testing",
            "test_output",
        )
        self.processor = self.get_processor()

    def get_processor(self):
        return song_processor.SongProcessor(
            self.input_path,
            self.output_path,
            8,
            6,
            "guitar",
            0.1
        )

    def test_notes(self):
        song_path = os.path.join(
            self.input_path,
            "notes.gp3"
        )
        chunks = self.processor.process_song(song_path, "train")
        expected = np.hstack(
            (
                np.diag(np.ones(12)),
                np.zeros((12, 1)),
                np.zeros((12, 7))

            )
        )
        expected[:, 15] = 1
        obtained = chunks[0]
        np.testing.assert_array_equal(expected, obtained)

    def test_rests(self):
        song_path = os.path.join(
            self.input_path,
            "rests.gp3"
        )
        obtained = self.processor.process_song(song_path, "train")
        self.assertIsNone(obtained)

    def test_note_dur(self):
        song_path = os.path.join(
            self.input_path,
            "notes_duration.gp3"
        )
        chunks = self.processor.process_song(song_path, "train")

        expected = np.hstack(
            (
                np.diag(np.ones(7)),
                np.zeros((7, 6)),
                np.diag(np.ones(7)),

            )
        )
        obtained = chunks[0]
        np.testing.assert_array_equal(expected, obtained)

    # TODO testing splitting, discontinuity and validation


class CustomSongProcessorTest(SongProcessorTest):
    def get_processor(self):
        return song_processor.CustomSongProcessor(
            self.input_path,
            self.output_path,
            8,
            6,
            "guitar",
            0.1,
        )
