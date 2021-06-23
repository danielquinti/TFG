#!/usr/bin/python3
import os
import re

import numpy as np
from libGPFile import *

string_note_map = {6: 4, 5: 9, 4: 2, 3: 7, 2: 11, 1: 4}

# string_note_map[0]=4
# string_note_map[1]=9
# string_note_map[2]=2
# string_note_map[3]=7
# string_note_map[4]=11
# string_note_map[5]=4


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


def find(l, condition=None):
    for idx, elem in enumerate(l):
        if condition(elem):
            return idx
    return None


def process_beat(beat):
    beat_vector = np.zeros(12)
    duration = 1 / (2 ** int.from_bytes(beat.duration, byteorder='big', signed=True))
    g_string = find(beat.strings, none_check)
    if g_string is None or beat.strings[g_string].noteType is None:
        beat_vector.fill(-1)
    else:
        base_note = string_to_base_note(g_string)
        offset = beat.strings[g_string].noteType[1]
        final_note = (base_note + offset) % 12
        beat_vector[final_note] = duration
    return beat_vector


def extract_sequence(beat_lists):
    pattern = []
    for measure in beat_lists:
        for beat in measure[0]:
            pattern.append(process_beat(beat))
    pattern = np.asarray(pattern)
    return pattern

def get_files(route):
    file_list = []
    for root, dirs, files in os.walk(route):
        for file in files:
            # append the file name to the list
            file_list.append(os.path.join(root, file))
    return file_list

if __name__ == "__main__":
    path = "../data"
    samples = get_files(path)
    for sample in samples:
        try:
            g = GPFile.read(sample)
        except EOFError:
            continue
        i = find(g.tracks, condition=track_name_match)
        if i is not None:
            g.dropAllTracksBut(i)
            sequence = extract_sequence(g.beatLists)