#!/usr/bin/python3
import csv
import json
import os
import shutil
from collections import defaultdict
from collections.abc import Callable
from typing import Any
import numpy as np
import song_processing.guitarpro as gp
import time


def get_file_paths(route):
    name_list = []
    for root, dirs, files in os.walk(route):
        for file in files:
            name_list.append(os.path.join(root, file))
    return name_list


def clean_tabs():
    def dump_merged_files(files_dict):
        os.makedirs(os.path.join("data", "clean_tabs", "gp3"))
        os.makedirs(os.path.join("data", "clean_tabs", "gp4"))
        os.makedirs(os.path.join("data", "clean_tabs", "gp5"))
        os.makedirs(os.path.join("data", "clean_tabs", "gtp"))

        for idx, full_path in enumerate(files_dict.values()):
            shutil.copy(
                full_path,
                os.path.join(
                    "data",
                    "clean_tabs",
                    os.path.splitext(os.path.basename(full_path))[1][1:],
                    os.path.basename(full_path)
                )
            )

    def get_unique_file_dict(path_list):
        def is_acceptable(song_path, fmt):
            if fmt not in ('.gp3', '.gp4', '.gp5', ".gtp") \
                    or os.path.getsize(song_path) <= 1024:
                return False
            return True

        files_dict = {}
        for filepath in path_list:
            # avoid duplicates
            name = os.path.basename(filepath)
            k, extension = os.path.splitext(name)
            if not is_acceptable(filepath, extension):
                continue
            if not files_dict.get(k):
                files_dict[k] = filepath
        return files_dict

    filepaths_1 = get_file_paths(
        os.path.join(
            "data",
            "60000Tabs"
        )
    )
    filepaths_2 = get_file_paths(
        os.path.join(
            "data",
            "Guitar_Pro_Tabs"
        )
    )
    print("Files in folder 1: {}".format(len(filepaths_1)))
    print("Files in folder 2: {}".format(len(filepaths_2)))
    files_dict_1 = get_unique_file_dict(filepaths_1)
    files_dict_2 = get_unique_file_dict(filepaths_2)
    print("Unique files in folder 1: {}".format(len(files_dict_1)))
    print("Unique files in folder 2: {}".format(len(files_dict_2)))
    intersection = set(files_dict_1.keys()).intersection(set(files_dict_2.keys()))
    print("Intersection: {}".format(len(intersection)))
    union = set(files_dict_1.keys()).union(set(files_dict_2.keys()))
    print("Union: {}".format(len(union)))
    difference = set(files_dict_1.keys()).difference(set(files_dict_2.keys()))
    print("Difference: {}".format(len(difference)))

    files_dict_merged = files_dict_1.copy()
    for key, value in files_dict_2.items():
        if key not in files_dict_merged:
            files_dict_merged[key] = value

    print("Unique files after merging: {}".format(len(files_dict_merged)))

    dump_merged_files(files_dict_merged)


def select_juicy_tracks():
    with open("track_names.csv", 'r') as f:
        reader = csv.reader(f)
        mydict = {rows[0]: rows[1] for rows in reader}

    total_guitar = sum(int(freq) for name, freq in mydict.items() if 'guitar' in name.lower())
    total_piano = sum(int(freq) for name, freq in mydict.items() if 'piano' in name.lower())
    total_bass = sum(int(freq) for name, freq in mydict.items() if 'bass' in name.lower())
    print("Total guitar:", total_guitar, "Total piano:", total_piano, "Total bass:", total_bass)


def __find_match_idx__(lst: list, condition: Callable[Any, bool]):
    for idx, elem in enumerate(lst):
        if condition(elem):
            return idx
    return None


class FileParsingException:
    pass


class SimpleBeat:
    def __init__(self, beat):
        self.beat = beat

    def is_chord(self):
        return len(self.beat.notes) > 1

    def is_rest(self):
        return not self.beat.notes

    def get_encoding(self):
        if self.is_chord():
            raise FileParsingException
        duration = 1 / self.beat.duration.value
        if self.beat.duration.isDotted:
            duration = duration * 1.5
        if self.is_rest():
            note = -1
        else:
            note = self.beat.notes[0].realValue
        return [note, duration]


def open_song(song_path):
    try:
        return gp.parse(song_path)
    except gp.GPException:
        return None


class SimpleTrack:
    def __init__(self, track):
        self.track = track

    def process(self, beat_thr, rest_thr):
        if self.track.isPercussionTrack:
            return None
        rest_acc = []
        current_chunk = []
        chunks = []
        for measure in self.track.measures:
            for beat in measure.voices[0].beats:
                beat = SimpleBeat(beat)
                if beat.is_chord():
                    # discard tracks with cords
                    return None
                if beat.is_rest():
                    rest_acc.append(beat.get_encoding())
                # the beat has a single note
                elif len(rest_acc) > rest_thr:  # the new note is from a different chunk
                    if len(current_chunk) >= beat_thr:  # avoid dumping sequences that are too short
                        chunks.append(np.array(current_chunk))
                    # reset accumulators and parse the current note
                    rest_acc.clear()
                    current_chunk = [beat.get_encoding()]

                elif len(rest_acc):  # new note within the same chunk after a sequence of rests
                    [current_chunk.append(x) for x in rest_acc]  # update accumulator with rest sequence
                    rest_acc.clear()
                    # parse and compute current note
                    current_chunk.append(beat.get_encoding())

                else:  # new note with no leading rests
                    current_chunk.append(beat.get_encoding())

        # dump the last notes of the file if the sequence is long enough
        if len(current_chunk) >= beat_thr:
            chunks.append(np.array(current_chunk))
        if chunks:
            return chunks
        else:
            return None


class SongProcessor:

    def __init__(
            self,
            in_path: str,
            out_path: str,
            rest_thr: int,
            beat_thr: int,
    ):
        self.input_path = in_path
        self.file_paths = get_file_paths(self.input_path)
        self.output_path = out_path
        self.rest_thr = rest_thr
        self.beat_thr = beat_thr
        self.note_range = 13
        self.duration_range = 7
        self.note_mod = self.note_range - 1
        self.beat_range = self.note_range + self.duration_range

    def get_track_insight(self):
        losses = 0
        wins = 0
        track_names = defaultdict(lambda: 0)
        track_tunings = defaultdict(lambda: 0)
        for idx, song_path in enumerate(self.file_paths):
            if idx >= 100:
                break
            try:
                song = gp.parse(song_path)
            except gp.GPException:
                losses += 1
                continue
            for track in song.tracks:
                key = track.name.lower()
                if key not in track_names:
                    track_names[key] = 1
                else:
                    track_names[key] += 1
            print("Losses:", losses, "Wins:", wins, " Attempts:", idx, " Tracks:", len(track_names))
            wins += 1

        with open('track_names.csv', 'w') as f:
            for key, value in track_names.items():
                f.write("%s, %s\n" % (key, value))
        with open('track_tunings.csv', 'w') as f:
            for key, value in track_tunings.items():
                f.write("%s, %s\n" % (key, value))

    def process_songs(self):
        for i, filepath in enumerate(self.file_paths):
            print(
                f'Split attempt: {i + 1}/{len(self.file_paths)}')
            self.process_song(filepath, i)

    def process_song(self, song_path: str, song_index: int):
        song = open_song(song_path)
        if song is None:
            return None
        song_name = os.path.basename(song_path).split('.')[0]
        for track_index, track in enumerate(song.tracks):
            track = SimpleTrack(track)
            chunks = track.process(self.beat_thr, self.rest_thr)
            if chunks:
                self.save_chunks(chunks, song_name, song_index, track_index)

    def save_chunks(
            self,
            chunks: list,
            song_name: str,
            song_number: int,
            track_number: int
    ):

        song_name = song_name.lower().lstrip().rstrip()
        target_folder = os.path.join(self.output_path, str(song_number), str(track_number))
        os.makedirs(target_folder, exist_ok=True)
        for idx, chunk in enumerate(chunks):
            chunk_name = f"{song_name}({track_number})({idx}).npy"
            destination = os.path.join(
                target_folder,
                chunk_name
            )
            np.save(destination, chunk)
