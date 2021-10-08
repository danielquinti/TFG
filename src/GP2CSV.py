#!/usr/bin/python3
import math
import os
import random
import re
import numpy as np

from libGPFile import *
from utils import *
from configparser import ConfigParser


class GP2CSV:
    def __init__(self):
        config = ConfigParser()
        config.read('.\config.ini')
        config_group = "GP2CSV"
        self.silence_thr = int(config.get(config_group, 'max_silence_threshold'))
        self.min_beats = int(config.get(config_group, 'minimum_beats'))
        self.input_path = config.get(config_group, 'input_path')
        self.output_path = config.get(config_group, 'output_path')
        self.test_rate = float(config.get(config_group, 'gp_test_rate'))
        self.track_name = config.get(config_group, 'track_name')
        self.string_to_base_note = {6: 4, 5: 9, 4: 2, 3: 7, 2: 11, 1: 4}
        self.gp_to_csv()

    def process_song(self, beat_lists, general_path):
        rest_ctr = 0
        part = 0
        contents = []
        for measure in beat_lists:
            for beat in measure[0]:
                beat_vector = np.zeros(13)
                # encode duration as the (2**n)th part of a beat
                duration = 1 / (2 ** (int.from_bytes(beat.duration, byteorder='big', signed=True) + 2))
                if sum(x is not None for x in beat.strings) > 1:  # chord
                    # discard tracks with cords
                    return
                g_string = find(beat.strings, lambda x: x is not None)
                if g_string is None or beat.strings[g_string].noteType is None:  # rest
                    rest_ctr += 1
                # the beat has a single note
                elif rest_ctr > self.silence_thr:  # the new note is from a different sample group
                    if len(contents) >= self.min_beats:  # avoid dumping sequences that are too short
                        final_path = general_path + "-" + str(part) + ".csv"
                        np.savetxt(final_path, np.asarray(contents), fmt='%1.6f')
                        part += 1

                    # reset accumulators and parse the current note
                    rest_ctr = 0
                    base_note = self.string_to_base_note[g_string]
                    offset = beat.strings[g_string].noteType[1]
                    note = (base_note + offset) % 12
                    beat_vector[note] = duration
                    contents = [beat_vector]

                elif rest_ctr > 0:  # new note within the same sample group after a sequence of rests
                    beat_vector[-1] = duration
                    [contents.append(beat_vector) for _ in range(rest_ctr)]  # update accumulator with rest sequence

                    # parse and compute current note
                    base_note = self.string_to_base_note[g_string]
                    offset = beat.strings[g_string].noteType[1]
                    note = (base_note + offset) % 12

                    # update accumulator with current note
                    beat_vector = np.zeros(13)
                    beat_vector[note] = duration
                    contents.append(beat_vector)
                    rest_ctr = 0
                else:  # new note with no leading rests
                    base_note = self.string_to_base_note[g_string]
                    offset = beat.strings[g_string].noteType[1]
                    note = (base_note + offset) % 12
                    beat_vector[note] = duration
                    contents.append(beat_vector)

        # dump the last notes of the file
        if len(contents) >= self.min_beats:
            final_path = general_path + "-" + str(part) + ".csv"
            np.savetxt(final_path, np.asarray(contents), fmt='%1.6f')

    def process_songs(self, file_paths, distribution):
        distribution_folder_path = os.path.join(self.output_path, distribution)
        if not os.path.exists(os.path.join(distribution_folder_path)):
            os.mkdir(distribution_folder_path)
        for i, file_path in enumerate(file_paths):
            print(i)
            # ignore unparsable files
            try:
                g = GPFile.read(file_path)
            except EOFError:
                continue
            # isolate, process and save guitar track
            track = find(g.tracks, lambda x: re.search(self.track_name, x.name, re.IGNORECASE))
            if track is not None:
                g.dropAllTracksBut(track)
                short_name = file_path.split("\\")[-1].split(".")[0]
                output_file_path = os.path.join(self.output_path, distribution, short_name)
                self.process_song(g.beatLists, output_file_path)

    def gp_to_csv(self):
        file_paths = get_file_paths(self.input_path)
        # GPFile-level shuffle
        file_paths = random.sample(file_paths, len(file_paths))
        # GPFile-level split
        border = math.floor(len(file_paths) * (1 - self.test_rate))
        train_file_names = file_paths[:border]
        test_file_names = file_paths[border:]
        self.process_songs(test_file_names, 'test')
        self.process_songs(train_file_names, 'train')

if __name__ == "__main__":
    GP2CSV().gp_to_csv()