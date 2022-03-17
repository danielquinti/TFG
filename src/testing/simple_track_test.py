import os
import unittest
import numpy as np
import sys

sys.path.append(
    os.path.join(
        "src",
        "song_processing"
    )
)
from src.song_processing import song_processor as sp


class SimpleTrackTest(unittest.TestCase):
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
        return sp.SongProcessor(
            self.input_path,
            self.output_path,
            8,
            6
        )

    def test_notes(self):
        song_path = os.path.join(
            self.input_path,
            "notes.gp3"
        )
        song = sp.open_song(song_path)
        track = sp.SimpleTrack(song.tracks[0])
        chunks = track.process(self.processor.beat_thr, self.processor.rest_thr)
        obtained = chunks[0]
        notes = np.arange(12).reshape(-1,1) + 48
        dur = np.zeros((12,1))
        dur[:]=0.25
        expected = np.hstack((notes,dur))
        np.array_equal(obtained,expected)

    def test_rests(self):
        song_path = os.path.join(
            self.input_path,
            "rests.gp3"
        )
        song = sp.open_song(song_path)
        track = sp.SimpleTrack(song.tracks[0])
        chunks = track.process(self.processor.beat_thr, self.processor.rest_thr)
        self.assertIsNone(chunks)

    def test_chords(self):
        song_path = os.path.join(
            self.input_path,
            "chord.gp3"
        )
        song = sp.open_song(song_path)
        track = sp.SimpleTrack(song.tracks[0])
        chunks = track.process(self.processor.beat_thr, self.processor.rest_thr)
        self.assertIsNone(chunks)


    def test_note_dur(self):
        song_path = os.path.join(
            self.input_path,
            "notes_duration.gp3"
        )
        song = sp.open_song(song_path)
        track = sp.SimpleTrack(song.tracks[0])
        chunks = track.process(self.processor.beat_thr, self.processor.rest_thr)
        obtained = chunks[0]
        notes = np.arange(7).reshape(-1,1) + 48
        indices = np.arange(7).reshape(-1,1)
        dur = 1/np.power(2,indices)
        expected = np.hstack((notes,dur))
        np.array_equal(obtained,expected)

    def test_multiple_chunks(self):
        song_path = os.path.join(
            self.input_path,
            "multiple_chunks.gp3"
        )
        song = sp.open_song(song_path)
        track = sp.SimpleTrack(song.tracks[0])
        chunks = track.process(self.processor.beat_thr, self.processor.rest_thr)
        self.assertEqual(len(chunks), 2)

    def test_single_chunk(self):
        song_path = os.path.join(
            self.input_path,
            "single_chunk.gp3"
        )
        song = sp.open_song(song_path)
        track = sp.SimpleTrack(song.tracks[0])
        chunks = track.process(self.processor.beat_thr, self.processor.rest_thr)
        self.assertEqual(len(chunks), 1)

    def test_too_short(self):
        song_path = os.path.join(
            self.input_path,
            "too_short.gp3"
        )
        song = sp.open_song(song_path)
        track = sp.SimpleTrack(song.tracks[0])
        chunks = track.process(self.processor.beat_thr, self.processor.rest_thr)
        self.assertEqual(len(chunks), 1)

    def test_hanging_rests(self):
        song_path = os.path.join(
            self.input_path,
            "hanging_rests.gp3"
        )
        song = sp.open_song(song_path)
        track = sp.SimpleTrack(song.tracks[0])
        obtained = track.process(self.processor.beat_thr, self.processor.rest_thr)[0]
        self.assertEqual(obtained.shape[0], 31)