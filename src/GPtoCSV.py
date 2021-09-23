#!/usr/bin/python3
import math
import os
import random
import re

import numpy as np

from libGPFile import *

SILENCE_THRESHOLD = 8
MINIMUM_TIMESTEPS = 6
source_path = "C:\\Users\\danie\\Downloads\\60000Tabs"
string_note_map = {6: 4, 5: 9, 4: 2, 3: 7, 2: 11, 1: 4}
test_rate = 0.1
note_number_map = {0: "C", 1: "C#", 2: "D", 3: "D#", 4: "E", 5: "F", 6: "F#", 7: "G", 8: "G#", 9: "A", 10: "A#",
                   11: "B"}


def string_to_base_note(g_str):
    return string_note_map[g_str]


def number_to_note(num):
    return note_number_map[num]


def track_name_match(elem):
    return re.search("guitar", elem.name, re.IGNORECASE)


def none_check(elem):
    return elem is not None


def find(lst, condition=None):
    for idx, elem in enumerate(lst):
        if condition(elem):
            return idx
    return None


def process_song(beat_lists, song_name):
    silence = 0
    part = 0
    contents = []
    for measure in beat_lists:
        for beat in measure[0]:
            beat_vector = np.zeros(13)
            duration = 1 / (2 ** (int.from_bytes(beat.duration, byteorder='big', signed=True) + 2))
            if sum(x is not None for x in beat.strings) > 1:  # chord
                return
            g_string = find(beat.strings, none_check)
            if g_string is None or beat.strings[g_string].noteType is None:  # rest
                silence += 1
            elif silence > SILENCE_THRESHOLD:
                silence = 0
                if len(contents) >= MINIMUM_TIMESTEPS:
                    np.savetxt((song_name + "-" + str(part) + ".csv"), np.asarray(contents), fmt='%1.6f')
                    part += 1

                base_note = string_to_base_note(g_string)
                offset = beat.strings[g_string].noteType[1]
                note = (base_note + offset) % 12
                beat_vector[note] = duration
                contents = [beat_vector]

            elif silence > 0:
                beat_vector[-1] = duration
                [contents.append(beat_vector) for _ in range(silence)]
                base_note = string_to_base_note(g_string)
                offset = beat.strings[g_string].noteType[1]
                note = (base_note + offset) % 12
                beat_vector = np.zeros(13)
                beat_vector[note] = duration
                contents.append(beat_vector)
                silence = 0
            else:
                base_note = string_to_base_note(g_string)
                offset = beat.strings[g_string].noteType[1]
                note = (base_note + offset) % 12
                beat_vector[note] = duration
                contents.append(beat_vector)
    if len(contents) >= MINIMUM_TIMESTEPS:
        np.savetxt(song_name + "-" + str(part) + ".csv", np.asarray(contents), fmt='%1.6f')


def get_files(route):
    file_list = []
    for root, dirs, files in os.walk(route):
        for file in files:
            file_list.append(os.path.join(root, file))
    return file_list


def convert():
    file_names = get_files(source_path)
    # GPFile-level shuffle
    file_names = random.sample(file_names, len(file_names))
    # GPFile-level split
    train_file_names = file_names[:math.floor(len(file_names) * (1 - test_rate))]
    test_file_names = file_names[math.floor(len(file_names) * (1 - test_rate)):]
    os.chdir("data\\test")
    process_songs(test_file_names)
    os.chdir("..\\train")
    process_songs(train_file_names)


def process_songs(name_list):
    for i, file_name in enumerate(name_list):
        print(i)
        try:
            g = GPFile.read(file_name)
        except EOFError:
            continue
        track = find(g.tracks, condition=track_name_match)
        if track is not None:
            g.dropAllTracksBut(track)
            process_song(g.beatLists, file_name.split("\\")[-1].split(".")[0])


if __name__ == "__main__":
    convert()
