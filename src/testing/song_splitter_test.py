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
from src.data_processing import song_splitter


class SongSplitterTest(unittest.TestCase):
    def test_notes(self):
        chunks = song_splitter.__check_and_split_song__(
            os.path.join(
                os.getcwd(),
                "src",
                "testing",
                "test_input",
                "notes.gp3"
            ),
            os.path.join(
                os.getcwd(),
                "src",
                "testing",
                "test_output",
            ),
            8,
            6,
            "guitar",
        )
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
        obtained = song_splitter.__check_and_split_song__(
            os.path.join(
                os.getcwd(),
                "src",
                "testing",
                "test_input",
                "rests.gp3"
            ),
            os.path.join(
                os.getcwd(),
                "src",
                "testing",
                "test_output",
            ),
            8,
            6,
            "guitar",
        )
        self.assertIsNone(obtained)

    def test_note_dur(self):
        chunks = song_splitter.__check_and_split_song__(
            os.path.join(
                os.getcwd(),
                "src",
                "testing",
                "test_input",
                "notes_duration.gp3"
            ),
            os.path.join(
                os.getcwd(),
                "src",
                "testing",
                "test_output",
            ),
            8,
            6,
            "guitar",
        )

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
