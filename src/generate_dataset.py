#!/usr/bin/python3
import os
import re
import time
import numpy as np
import random
import math
from libGPFile import *

GOOD_SONGS = 6752
END_OF_SONG = np.full(12, -2)
N = 6
source_path = "data\\tabs"
string_note_map = {6: 4, 5: 9, 4: 2, 3: 7, 2: 11, 1: 4}

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


def extract_sequence(beat_lists):
    pattern = []
    chords = 0
    silence = False
    for measure in beat_lists:
        for beat in measure[0]:
            beat_vector = np.zeros(12)
            duration = 1 / (2 ** (int.from_bytes(beat.duration, byteorder='big', signed=True) + 2))
            # TODO change to normal list search
            g_string = find(beat.strings, none_check)
            if sum(x is not None for x in beat.strings) >= 1:
                chords += 1
            if g_string is None or beat.strings[g_string].noteType is None:
                if not silence:
                    silence = True
                    beat_vector.fill(-1)
                    pattern.append(beat_vector)
            else:
                silence = False
                base_note = string_to_base_note(g_string)
                offset = beat.strings[g_string].noteType[1]
                final_note = (base_note + offset) % 12
                beat_vector[final_note] = duration
                pattern.append(beat_vector)
    pattern = np.asarray(pattern)
    return pattern, chords


def get_files(route):
    file_list = []
    for root, dirs, files in os.walk(route):
        for file in files:
            file_list.append(os.path.join(root, file))
    return file_list


def fill_lists(sequence, chords, sequence_list, chords_list):
    [sequence_list.append(bt) for bt in sequence[-N:]]
    # sequence_list.append(END_OF_SONG)
    chords_list.append(chords)


def generate_and_split():
    train = []
    dev = []
    test = []
    train_chords = []
    dev_chords = []
    test_chords = []
    dataset = []
    samples = get_files(source_path)
    random.shuffle(samples)
    counter = 0
    start_time = time.time()
    for sample in samples:
        counter += 1
        print(counter)
        try:
            g = GPFile.read(sample)
        except EOFError:
            continue
        track = find(g.tracks, condition=track_name_match)
        if track is not None:
            g.dropAllTracksBut(track)
            dataset.append(extract_sequence(g.beatLists))

    [fill_lists(sequence, chords, train, train_chords) for sequence, chords in
     random.sample(dataset[:math.floor(GOOD_SONGS * 0.9)], math.floor(GOOD_SONGS * 0.80))]
    [fill_lists(sequence, chords, dev, dev_chords) for sequence, chords in
     random.sample(dataset[:math.floor(GOOD_SONGS * 0.9)], math.floor(GOOD_SONGS * 0.10))]
    [fill_lists(sequence, chords, test, test_chords) for sequence, chords in
     random.sample(dataset[math.floor(GOOD_SONGS * 0.9):], math.floor(GOOD_SONGS * 0.10))]

    train = np.vstack(train)
    dev = np.vstack(dev)
    test = np.vstack(test)
    train_chords = np.array(train_chords)
    dev_chords = np.array(dev_chords)
    test_chords = np.array(test_chords)
    os.chdir("data")
    np.savetxt("train.csv", train, delimiter=",", fmt='%1.6f')
    np.savetxt("dev.csv", dev, delimiter=",", fmt='%1.6f')
    np.savetxt("test.csv", test, delimiter=",", fmt='%1.6f')
    np.savetxt("train_chords.csv", train_chords, delimiter=",", fmt='%1.6f')
    np.savetxt("dev_chords.csv", dev_chords, delimiter=",", fmt='%1.6f')
    np.savetxt("test_chords.csv", test_chords, delimiter=",", fmt='%1.6f')
    print("--- %s seconds ---" % (time.time() - start_time))


if __name__ == "__main__":
    generate_and_split()
