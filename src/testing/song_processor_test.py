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
        chunks = self.processor.process_song(song_path)
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
        obtained = self.processor.process_song(song_path)
        self.assertIsNone(obtained)

    def test_chords(self):
        song_path = os.path.join(
            self.input_path,
            "chord.gp3"
        )
        song = self.processor.read_song(song_path)
        track = self.processor.get_valid_track(song)
        measure = self.processor.get_measure_list(track)[0]
        beat = self.processor.get_beat_list(measure)[0]
        self.assertTrue(self.processor.is_chord(beat))

    def test_note_dur(self):
        song_path = os.path.join(
            self.input_path,
            "notes_duration.gp3"
        )
        chunks = self.processor.process_song(song_path)

        expected = np.hstack(
            (
                np.diag(np.ones(7)),
                np.zeros((7, 6)),
                np.diag(np.ones(7)),

            )
        )
        obtained = chunks[0]
        np.testing.assert_array_equal(expected, obtained)

    def test_multiple_chunks(self):
        song_path = os.path.join(
            self.input_path,
            "multiple_chunks.gp3"
        )
        chunks = self.processor.process_song(song_path)
        expected_0 = np.zeros((16, 20))
        expected_0[:, 0] = 1
        expected_0[:, 16] = 1

        expected_1 = np.zeros((8, 20))
        expected_1[:, 0] = 1
        expected_1[:, 16] = 1

        obtained_0 = chunks[0]
        obtained_1 = chunks[1]
        np.testing.assert_array_equal(expected_0, obtained_0)
        np.testing.assert_array_equal(expected_1, obtained_1)

    def test_single_chunk(self):
        song_path = os.path.join(
            self.input_path,
            "single_chunk.gp3"
        )
        chunks = self.processor.process_song(song_path)
        rests = np.zeros((8, 20))
        notes = rests.copy()
        rests[:, 12] = 1
        rests[:, 15] = 1
        notes[:, 0] = 1
        notes[:, 16] = 1
        expected = np.vstack(
            (
                notes,
                notes,
                rests,
                notes

            )
        )

        obtained = chunks[0]
        np.testing.assert_array_equal(expected, obtained)

    def test_too_short(self):
        song_path = os.path.join(
            self.input_path,
            "too_short.gp3"
        )
        chunks = self.processor.process_song(song_path)
        self.assertEqual(1, len(chunks))

    def test_hanging_rests(self):
        song_path = os.path.join(
            self.input_path,
            "hanging_rests.gp3"
        )
        chunks = self.processor.process_song(song_path)
        rests = np.zeros((8, 20))
        notes_0 = rests.copy()
        rests[:, 12] = 1
        rests[:, 15] = 1
        notes_0[:, 0] = 1
        notes_0[:, 16] = 1
        notes_1 = np.zeros((7, 20))
        notes_1[:, 0] = 1
        notes_1[:, 16] = 1
        expected = np.vstack(
            (
                notes_0,
                notes_0,
                rests,
                notes_1

            )
        )
        obtained = chunks[0]
        np.testing.assert_array_equal(expected, obtained)

    def test_multiple_tracks(self):
        song_path = os.path.join(
            self.input_path,
            "multiple_tracks.gp3"
        )
        chunks = self.processor.process_song(song_path)
        expected = np.zeros((6, 20))
        expected[:, 0] = 1
        expected[:, 13] = 1

        obtained = chunks[0]
        np.testing.assert_array_equal(expected, obtained)


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
