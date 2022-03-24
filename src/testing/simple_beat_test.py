import os
import unittest
import numpy as np
import sys

sys.path.append(
    os.path.join(
        "..",
        "src",
        "song_processing"
    )
)
from src.song_processing import song_processor as sp


class SimpleBeatTest(unittest.TestCase):
    def setUp(self):
        self.input_path = os.path.join(
            os.getcwd(),
            "testing",
            "test_input"
        )
        self.output_path = os.path.join(
            os.getcwd(),
            "testing",
            "test_output",
        )
        self.processor = self.get_processor()

    def get_processor(self):
        return sp.SongProcessor(
            self.input_path,
            self.output_path,
            8,
            6,
            1000
        )

    def test_notes(self):
        song_path = os.path.join(
            self.input_path,
            "notes.gp3"
        )
        song = sp.open_song(song_path)
        track = sp.SimpleTrack(song.tracks[0])
        chunks = track.process(self.processor.min_beat_thr, self.processor.max_beat_thr, self.processor.rest_thr)
        obtained = [beat[0] for beat in chunks[0]]
        expected = np.arange(12).reshape(-1,1) + 48
        np.array_equal(obtained,expected)


    def test_duration(self):
        song_path = os.path.join(
            self.input_path,
            "notes_duration.gp3"
        )
        song = sp.open_song(song_path)
        track = sp.SimpleTrack(song.tracks[0])
        chunks = track.process(self.processor.min_beat_thr, self.processor.max_beat_thr, self.processor.rest_thr)
        obtained = chunks[0]
        expected = np.arange(7).reshape(-1,1)
        np.array_equal(obtained,expected)

    def test_chord(self):
        song_path = os.path.join(
            self.input_path,
            "chord.gp3"
        )
        song = sp.open_song(song_path)
        track = song.tracks[0]
        for measure in track.measures:
            for beat in measure.voices[0].beats:
                beat = sp.SimpleBeat(beat)
                self.assertTrue(beat.is_chord)
                break

    def test_dotted(self):
        song_path = os.path.join(
            self.input_path,
            "dotted.gp3"
        )
        song = sp.open_song(song_path)
        track = song.tracks[0]
        for measure in track.measures:
            for beat in measure.voices[0].beats:
                beat = sp.SimpleBeat(beat)
                self.assertTrue(beat.is_dotted)
                break

    def test_encoding(self):
        song_path = os.path.join(
            self.input_path,
            "dotted.gp3"
        )
        song = sp.open_song(song_path)
        track = song.tracks[0]
        for measure in track.measures:
            for beat in measure.voices[0].beats:
                beat = sp.SimpleBeat(beat)
                obtained = beat.encoding
                expected = np.array([0, 3, 2, 1])
                np.array_equal(obtained, expected)
                break
