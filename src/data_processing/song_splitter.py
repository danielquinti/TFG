#!/usr/bin/python3
import json
import math
import os
import random
import re
from collections.abc import Callable
from typing import Any

import numpy as np
import libGPFile
import utils


def __save_chunks__(
        output_path: str,
        dist_name: str,
        song_name: str,
        chunks: list
):
    for idx, chunk in enumerate(chunks):
        chunk_name = f"{song_name}-{idx}"
        csv_name = os.path.join(
            output_path,
            "csv",
            dist_name,
            f"{chunk_name}.csv"
        )
        npy_name = os.path.join(
            output_path,
            "npy",
            dist_name,
            f"{chunk_name}.npy"
        )
        np.savetxt(csv_name, np.asarray(chunk), fmt="%d")
        np.save(npy_name, np.asarray(chunk))


def __split_song__(beat_lists: list[libGPFile.GPFile.GPMeasure],
                   silence_thr: int,
                   min_beats: int):
    string_to_base_note = {6: 4, 5: 9, 4: 2, 3: 7, 2: 11, 1: 4}
    rest_ctr = 0
    rest_acc = []
    note_range = 13
    note_mod = note_range - 1
    duration_range = 7
    beat_range = note_range + duration_range
    contents = []
    chunks = []
    for measure in beat_lists:
        for beat in measure[0]:
            beat_vector = np.zeros(beat_range)
            duration = int.from_bytes(beat.duration, byteorder='big', signed=True) + 2

            if sum(x is not None for x in beat.strings) > 1:  # chord
                # discard tracks with cords
                return

            g_string = __find_match__(beat.strings, lambda x: x is not None)
            if g_string is None or beat.strings[g_string].noteType is None:  # rest
                rest_ctr += 1
                # toggle the indexes of the last note (rest) and the corresponding duration
                beat_vector[note_mod] = 1
                beat_vector[note_range + duration] = 1

                rest_acc.append(beat_vector)
            # the beat has a single note
            elif rest_ctr > silence_thr:  # the new note is from a different sample group
                if len(contents) >= min_beats:  # avoid dumping sequences that are too short
                    chunks.append(np.array(contents))
                # reset accumulators and parse the current note
                rest_ctr = 0
                rest_acc = []
                base_note = string_to_base_note[g_string]
                offset = beat.strings[g_string].noteType[1]
                note = (base_note + offset) % note_mod
                # toggle the indexes of the last corresponding note and duration
                beat_vector[note] = 1
                beat_vector[note_range + duration] = 1

                contents = [beat_vector]

            elif rest_ctr > 0:  # new note within the same sample group after a sequence of rests
                [contents.append(x) for x in rest_acc]  # update accumulator with rest sequence
                # parse and compute current note
                base_note = string_to_base_note[g_string]
                offset = beat.strings[g_string].noteType[1]
                note = (base_note + offset) % 12

                # update accumulator with current note
                beat_vector = np.zeros(beat_range)
                beat_vector[note] = 1
                beat_vector[note_range + duration] = 1
                contents.append(beat_vector)
                rest_ctr = 0
                rest_acc = []
            else:  # new note with no leading rests
                base_note = string_to_base_note[g_string]
                offset = beat.strings[g_string].noteType[1]
                note = (base_note + offset) % 12
                # toggle the indexes of the last corresponding note and duration
                beat_vector[note] = 1
                beat_vector[note_range + duration] = 1

                contents.append(beat_vector)

    # dump the last notes of the file
    if len(contents) >= min_beats:
        chunks.append(np.array(contents))
    return chunks


def __find_match__(lst: list, condition: Callable[Any, bool]):
    for idx, elem in enumerate(lst):
        if condition(elem):
            return idx
    return None


def __check_and_split_song__(
        input_path: str,
        output_path: str,
        dist_name: str,
        silence_thr: int,
        min_beats: int,
        track_name: str):
    # ignore unparsable files
    try:
        g = libGPFile.GPFile.read(input_path)
    except EOFError:
        return
    # isolate, process and save guitar track
    track = __find_match__(g.tracks, lambda x: re.search(track_name, x.name, re.IGNORECASE))
    if track is not None:
        g.dropAllTracksBut(track)

        chunks = __split_song__(
            g.beatLists,
            silence_thr,
            min_beats
        )
        song_name = input_path.split("\\")[-1].split(".")[0]
        if chunks:
            __save_chunks__(
                output_path,
                dist_name,
                song_name,
                chunks
            )
            return chunks
        else:
            return None


def __split_songs__(
        input_path: str,
        output_path: str,
        silence_thr: int,
        min_beats: int,
        track_name: str,
        test_rate: float
):
    file_paths = utils.get_file_paths(input_path)
    # GPFile-level shuffle
    file_paths = random.sample(file_paths, len(file_paths))
    # GPFile-level split

    border = math.floor(len(file_paths) * (1 - test_rate))
    dist_paths = {
        "train": file_paths[:border],
        "test": file_paths[border:]
    }
    os.makedirs(
        os.path.join(
            output_path,
            "csv",
            "train",
        ),
        exist_ok=True
    )
    os.makedirs(
        os.path.join(
            output_path,
            "csv",
            "test",
        ),
        exist_ok=True
    )
    os.makedirs(
        os.path.join(
            output_path,
            "npy",
            "train",
        ),
        exist_ok=True
    )
    os.makedirs(
        os.path.join(
            output_path,
            "npy",
            "test",
        ),
        exist_ok=True
    )

    for dist_name, file_paths in dist_paths.items():
        for i, file_path in enumerate(file_paths):
            print(f'Split attempt: {i+1}/{len(file_paths)}')
            __check_and_split_song__(
                file_path,
                output_path,
                dist_name,
                silence_thr,
                min_beats,
                track_name
            )


def split_songs():
    with open(
            os.path.join(
                "src",
                "config",
                "song_splitter_config.json"
            )
    ) as fp:
        params = json.load(fp)

    input_path: str = os.path.join(*(params["input_path"].split("\\")[0].split("/")))
    output_path: str = os.path.join(*(params["output_path"].split("\\")[0].split("/")))
    track_name: str = params["track_name"]
    silence_thr: int = params["silence_thr"]
    min_beats: int = params["min_beats"]
    test_rate: float = params["test_rate"]

    __split_songs__(
        input_path,
        output_path,
        silence_thr,
        min_beats,
        track_name,
        test_rate
    )

if __name__ == "__main__":
    split_songs()