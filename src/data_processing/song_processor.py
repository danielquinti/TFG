#!/usr/bin/python3
import csv
import json
import math
import os
import random
import re
from collections import defaultdict
from collections.abc import Callable
from typing import Any

import numpy as np
import libGPFile
import PyGuitarPro.guitarpro as gp
import utils


def shuffle_and_split_files(file_paths, split_rate):
    file_paths = random.sample(file_paths, len(file_paths))
    border = math.floor(len(file_paths) * (1 - split_rate))
    return {
        "train": file_paths[:border],
        "dev-test": file_paths[border:]
    }


def __find_match__(lst: list, condition: Callable[Any, bool]):
    for idx, elem in enumerate(lst):
        if condition(elem):
            return idx
    return None


def prepare_dirs(destination):
    os.makedirs(
        os.path.join(
            destination,
            "csv",
            "train",
        ),
        exist_ok=True
    )
    os.makedirs(
        os.path.join(
            destination,
            "csv",
            "dev-test",
        ),
        exist_ok=True
    )
    os.makedirs(
        os.path.join(
            destination,
            "npy",
            "train",
        ),
        exist_ok=True
    )
    os.makedirs(
        os.path.join(
            destination,
            "npy",
            "dev-test",
        ),
        exist_ok=True
    )


def __save_chunks__(
        destination: str,
        dist_name: str,
        song_name: str,
        chunks: list
):
    for idx, chunk in enumerate(chunks):
        chunk_name = f"{song_name}-{idx}"
        csv_name = os.path.join(
            destination,
            "csv",
            dist_name,
            f"{chunk_name}.csv"
        )
        npy_name = os.path.join(
            destination,
            "npy",
            dist_name,
            f"{chunk_name}.npy"
        )
        np.savetxt(csv_name, np.asarray(chunk), fmt="%d")
        np.save(npy_name, np.asarray(chunk))


class SongProcessor:
    def __init__(
            self,
            in_path: str,
            out_path: str,
            rest_thr: int,
            beat_thr: int,
            wanted_track_name: str,
            split_rate: float
    ):
        self.input_path = in_path
        self.output_path = out_path
        self.rest_thr = rest_thr
        self.beat_thr = beat_thr
        self.track_name = wanted_track_name
        self.split_rate = split_rate

    def process_songs(self):
        file_paths = utils.get_file_paths(self.input_path)
        dist_paths = shuffle_and_split_files(file_paths, self.split_rate)
        prepare_dirs(self.output_path)
        for dist_name, file_paths in dist_paths.items():
            for i, filepath in enumerate(file_paths):
                print(f'Split attempt: {i + 1}/{len(file_paths)}')
                chunks = self.process_song(filepath, dist_name)
                song_name = self.input_path.split("\\")[-1].split(".")[0]
                if chunks:
                    __save_chunks__(
                        self.output_path,
                        dist_name,
                        song_name,
                        chunks
                    )

    def get_duration(self, beat):
        return beat.duration.index

    def is_chord(self, beat):
        return len(beat.notes) > 1

    def is_rest(self, beat):
        return not beat.notes

    def compute_note(self, beat):
        note=beat.notes[0]
        string_to_base_note = {6: 4, 5: 9, 4: 2, 3: 7, 2: 11, 1: 4}
        note_mod = 12
        g_string = note.string
        base_note = string_to_base_note[g_string]
        offset = note.value
        return (base_note + offset) % note_mod
        return 0
    def get_measure_list(self, track):
        return track.measures

    def get_beat_list(self, measure):
        return measure.voices[0].beats

    def process_track(
            self,
            track,
            rest_thr: int,
            beat_thr: int
    ):
        rest_ctr = 0
        rest_acc = []
        note_range = 13
        note_mod = note_range - 1
        duration_range = 7
        beat_range = note_range + duration_range
        contents = []
        chunks = []
        for measure in self.get_measure_list(track):
            for beat in self.get_beat_list(measure):
                beat_vector = np.zeros(beat_range)
                duration = self.get_duration(beat)

                if self.is_chord(beat):
                    # discard tracks with cords
                    return None

                if self.is_rest(beat):
                    rest_ctr += 1
                    # toggle the indexes of the last note (rest) and the corresponding duration
                    beat_vector[note_mod] = 1
                    beat_vector[note_range + duration] = 1

                    rest_acc.append(beat_vector)
                # the beat has a single note
                elif rest_ctr > rest_thr:  # the new note is from a different sample group
                    if len(contents) >= beat_thr:  # avoid dumping sequences that are too short
                        chunks.append(np.array(contents))
                    # reset accumulators and parse the current note
                    rest_ctr = 0
                    rest_acc = []
                    note = self.compute_note(beat)
                    # toggle the indexes of the last corresponding note and duration
                    beat_vector[note] = 1
                    beat_vector[note_range + duration] = 1

                    contents = [beat_vector]

                elif rest_ctr > 0:  # new note within the same sample group after a sequence of rests
                    [contents.append(x) for x in rest_acc]  # update accumulator with rest sequence
                    # parse and compute current note
                    note = self.compute_note(beat)


                    # update accumulator with current note
                    beat_vector = np.zeros(beat_range)
                    beat_vector[note] = 1
                    beat_vector[note_range + duration] = 1
                    contents.append(beat_vector)
                    rest_ctr = 0
                    rest_acc = []
                else:  # new note with no leading rests
                    note = self.compute_note(beat)
                    # toggle the indexes of the last corresponding note and duration
                    beat_vector[note] = 1
                    beat_vector[note_range + duration] = 1

                    contents.append(beat_vector)

        # dump the last notes of the file
        if len(contents) >= beat_thr:
            chunks.append(np.array(contents))
        return chunks

    def process_song(self, song_path, dist_name):
        song = self.safe_read_song(song_path)
        if song is not None:
            track_number = __find_match__(song.tracks, lambda x: re.search(self.track_name, x.name, re.IGNORECASE))
            if track_number is not None:
                track = song.tracks[track_number]
                chunks = self.process_track(track, self.rest_thr, self.beat_thr)
                if chunks:
                    return chunks
        return None

    def safe_read_song(self, filepath):
        try:
            return gp.parse(filepath)
        except gp.GPException:
            return None


class CustomSongProcessor(SongProcessor):
    def safe_read_song(self, filepath):
        try:
            return libGPFile.GPFile.read(filepath)
        except EOFError:
            return None

    def process_song(self, song_path, dist_name):
        # ignore unparsable files
        try:
            song = libGPFile.GPFile.read(song_path)
        except EOFError:
            return
        # isolate, process and save guitar track
        track = __find_match__(song.tracks, lambda x: re.search(self.track_name, x.name, re.IGNORECASE))
        if track is not None:
            song.dropAllTracksBut(track)
            chunks = self.process_track(song.beatLists, self.rest_thr, self.beat_thr)
            if chunks:
                return chunks
        return None

    def get_duration(self, beat):
        return int.from_bytes(beat.duration, byteorder='big', signed=True) + 2

    def is_chord(self, beat):
        return sum(x is not None for x in beat.strings) > 1

    def is_rest(self, beat):
        g_string = __find_match__(beat.strings, lambda x: x is not None)
        return g_string is None or beat.strings[g_string].noteType is None

    def compute_note(self, beat):
        string_to_base_note = {6: 4, 5: 9, 4: 2, 3: 7, 2: 11, 1: 4}
        note_mod = 12
        g_string = __find_match__(beat.strings, lambda x: x is not None)
        base_note = string_to_base_note[g_string]
        offset = beat.strings[g_string].noteType[1]
        return (base_note + offset) % note_mod

    def get_measure_list(self, track):
        return track


    def get_beat_list(self, measure):
        return measure[0]

    def process_track(self, track, rest_thr, beat_thr):
        rest_ctr = 0
        rest_acc = []
        note_range = 13
        note_mod = note_range - 1
        duration_range = 7
        beat_range = note_range + duration_range
        contents = []
        chunks = []
        for measure in self.get_measure_list(track):
            for beat in self.get_beat_list(measure):
                beat_vector = np.zeros(beat_range)
                duration = self.get_duration(beat)

                if self.is_chord(beat):
                    # discard tracks with cords
                    return None

                if self.is_rest(beat):
                    rest_ctr += 1
                    # toggle the indexes of the last note (rest) and the corresponding duration
                    beat_vector[note_mod] = 1
                    beat_vector[note_range + duration] = 1

                    rest_acc.append(beat_vector)
                # the beat has a single note
                elif rest_ctr > rest_thr:  # the new note is from a different sample group
                    if len(contents) >= beat_thr:  # avoid dumping sequences that are too short
                        chunks.append(np.array(contents))
                    # reset accumulators and parse the current note
                    rest_ctr = 0
                    rest_acc = []
                    note = self.compute_note(beat)
                    # toggle the indexes of the last corresponding note and duration
                    beat_vector[note] = 1
                    beat_vector[note_range + duration] = 1

                    contents = [beat_vector]

                elif rest_ctr > 0:  # new note within the same sample group after a sequence of rests
                    [contents.append(x) for x in rest_acc]  # update accumulator with rest sequence
                    # parse and compute current note
                    note = self.compute_note(beat)


                    # update accumulator with current note
                    beat_vector = np.zeros(beat_range)
                    beat_vector[note] = 1
                    beat_vector[note_range + duration] = 1
                    contents.append(beat_vector)
                    rest_ctr = 0
                    rest_acc = []
                else:  # new note with no leading rests
                    note = self.compute_note(beat)
                    # toggle the indexes of the last corresponding note and duration
                    beat_vector[note] = 1
                    beat_vector[note_range + duration] = 1

                    contents.append(beat_vector)

        # dump the last notes of the file
        if len(contents) >= beat_thr:
            chunks.append(np.array(contents))
        return chunks


if __name__ == "__main__":
    with open(
            os.path.join(
                "src",
                "config",
                "song_processor_config.json"
            )
    ) as fp:
        params = json.load(fp)

    input_path: str = os.path.join(*(params["input_path"].split("\\")[0].split("/")))

    output_path: str = os.path.join(*(params["output_path"].split("\\")[0].split("/")))
    prepare_dirs(output_path)
    track_name: str = params["track_name"]
    silence_thr: int = params["silence_thr"]
    min_beats: int = params["min_beats"]
    test_rate: float = params["test_rate"]
    parser_class_index = 1 if params["parser_class"] == "custom" else 0
    parser_classes=[SongProcessor,CustomSongProcessor]
    parser=parser_classes[parser_class_index](
        input_path,
        output_path,
        silence_thr,
        min_beats,
        track_name,
        test_rate
    )
    parser.process_songs()
