#!/usr/bin/python3
import os
import re
import time
import numpy as np
import json
import random
import math
from libGPFile import *

GOOD_SONGS = 6752
END_OF_SONG = np.full(12, -2)
N = 6
SILENCE_THRESHOLD = 50
MINIMUM_TIMESTEPS = 10
source_path = "C:\\Users\\danie\\Downloads\\60000Tabs"
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


def process_song(beat_lists, song_name):
    chords = 0
    silence = 0
    part = 0
    contents = []
    for measure in beat_lists:
        for beat in measure[0]:
            beat_vector = np.zeros(13)
            duration = 1 / (2 ** (int.from_bytes(beat.duration, byteorder='big', signed=True) + 2))
            # TODO change to normal list search
            g_string = find(beat.strings, none_check)  # acorde
            if sum(x is not None for x in beat.strings) >= 1:
                chords += 1
            if g_string is None or beat.strings[g_string].noteType is None:  # silencio
                silence += 1
            # TODO hay archivos con notas y silencios
            elif silence > SILENCE_THRESHOLD:
                if len(contents)>=MINIMUM_TIMESTEPS:
                    np.savetxt(song_name + "_" + str(part) + ".csv", np.asarray(contents), fmt='%1.6f')
                    chords_dict[song_name + "_" + str(part)] = chords
                    part += 1
                chords = 0

                base_note = string_to_base_note(g_string)
                offset = beat.strings[g_string].noteType[1]
                note = (base_note + offset) % 12
                beat_vector[note] = duration
                contents = [beat_vector]

            elif silence > 0:
                #TODO es duration o 1?
                beat_vector[-1]=duration
                [contents.append(beat_vector) for i in range(silence)]
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
    if len(contents)>=MINIMUM_TIMESTEPS:
        np.savetxt(song_name + "_" + str(part+1) + ".csv", np.asarray(contents), fmt='%1.6f')

def get_files(route):
    file_list = []
    for root, dirs, files in os.walk(route):
        for file in files:
            file_list.append(os.path.join(root, file))
    return file_list


def generate():
    file_names = get_files(source_path)
    os.chdir("data\\dump")
    counter=1
    for file_name in file_names:
        counter+=1
        print(counter)
        try:
            g = GPFile.read(file_name)
        except EOFError:
            continue
        track = find(g.tracks, condition=track_name_match)
        if track is not None:
            g.dropAllTracksBut(track)
            process_song(g.beatLists, file_name.split("\\")[-1].split(".")[0])
    os("..")
    with open('chords.json', 'w') as fp:
        json.dump(chords_dict, fp)


chords_dict = dict()
if __name__ == "__main__":
    generate()
