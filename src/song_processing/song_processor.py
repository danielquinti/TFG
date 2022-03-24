#!/usr/bin/python3
import csv
import os
import shutil
from collections import defaultdict
from collections.abc import Callable
from typing import Any
import numpy as np
import song_processing.guitarpro as gp


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


class FileParsingException(Exception):
    pass


class SimpleBeat:
    def __init__(self, beat):
        self.beat = beat
        self.is_chord = len(self.beat.notes) > 1
        self.is_rest = not self.beat.notes
        self.note, self.octave = self.get_pitch()
        self.duration = self.beat.duration.index
        self.is_dotted = self.beat.duration.isDotted
        self.encoding = self.get_encoding()

    def get_pitch(self):
        if self.is_rest:
            note = 12
            octave = 10
        else:
            real_note = self.beat.notes[0].realValue
            note = real_note % 12
            octave = (real_note // 12)
        return note, octave

    def get_encoding(self):
        return [self.note, self.octave, self.duration, self.is_dotted]


def open_song(song_path):
    try:
        return gp.parse(song_path)
    except:
        return None


class SimpleTrack:
    instrument_list = [
        "Acoustic Grand Piano",
        "Bright Acoustic Piano",
        "Electric Grand Piano",
        "Honky-tonk Piano",
        "Electric Piano 1 (usually a Rhodes Piano)",
        "Electric Piano 2 (usually an FM piano patch)",
        "Harpsichord",
        "Clavinet",
        "Celesta",
        "Glockenspiel",
        "Music Box",
        "Vibraphone",
        "Marimba",
        "Xylophone",
        "Tubular Bells",
        "Dulcimer",
        "Drawbar Organ",
        "Percussive Organ",
        "Rock Organ",
        "Church Organ",
        "Reed Organ",
        "Accordion",
        "Harmonica",
        "Tango Accordion",
        "Acoustic Guitar (nylon)",
        "Acoustic Guitar (steel)",
        "Electric Guitar (jazz)",
        "Electric Guitar (clean)",
        "Electric Guitar (muted)",
        "Electric Guitar (overdriven)",
        "Electric Guitar (distortion)",
        "Electric Guitar (harmonics)",
        "Acoustic Bass",
        "Electric Bass (finger)",
        "Electric Bass (picked)",
        "Fretless Bass",
        "Slap Bass 1",
        "Slap Bass 2",
        "Synth Bass 1",
        "Synth Bass 2",
        "Violin",
        "Viola",
        "Cello",
        "Contrabass",
        "Tremolo Strings",
        "Pizzicato Strings",
        "Orchestral Harp",
        "Timpani",
        "String Ensemble 1",
        "String Ensemble 2",
        "Synth Strings 1",
        "Synth Strings 2",
        "Choir Aahs",
        "Voice Oohs (or Doos)",
        "Synth Voice or Solo Vox",
        "Orchestra Hit",
        "Trumpet",
        "Trombone",
        "Tuba",
        "Muted Trumpet",
        "French Horn",
        "Brass Section",
        "Synth Brass 1",
        "Synth Brass 2",
        "Soprano Sax",
        "Alto Sax",
        "Tenor Sax",
        "Baritone Sax",
        "Oboe",
        "English Horn",
        "Bassoon",
        "Clarinet",
        "Piccolo",
        "Flute",
        "Recorder",
        "Pan Flute",
        "Blown bottle",
        "Shakuhachi",
        "Whistle",
        "Ocarina",
        "Lead 1 (square)",
        "Lead 2 (sawtooth)",
        "Lead 3 (calliope)",
        "Lead 4 (chiff)",
        "Lead 5 (charang, a guitar-like lead)",
        "Lead 6 (space voice)",
        "Lead 7 (fifths)",
        "Lead 8 (bass and lead)",
        "Pad 1 (new age or fantasia, a warm pad stacked with a bell)",
        "Pad 2 (warm)",
        "Pad 3 (polysynth or poly)",
        "Pad 4 (choir)",
        "Pad 5 (bowed glass or bowed)",
        "Pad 6 (metallic)",
        "Pad 7 (halo)",
        "Pad 8 (sweep)",
        "FX 1 (rain)",
        "FX 2 (soundtrack, a bright perfect fifth pad)",
        "FX 3 (crystal)",
        "FX 4 (atmosphere, usually a nylon-like sound)",
        "FX 5 (brightness)",
        "FX 6 (goblins)",
        "FX 7 (echoes or echo drops)",
        "FX 8 (sci-fi or star theme)",
        "Sitar",
        "Banjo",
        "Shamisen",
        "Koto",
        "Kalimba",
        "Bag pipe",
        "Fiddle",
        "Shanai",
        "Tinkle Bell",
        "Agogô",
        "Steel Drums",
        "Woodblock",
        "Taiko Drum",
        "Melodic Tom or 808 Toms",
        "Synth Drum",
        "Reverse Cymbal",
        "Guitar Fret Noise",
        "Breath Noise",
        "Seashore",
        "Bird Tweet",
        "Telephone Ring",
        "Helicopter",
        "Applause",
        "Gunshot"
    ]

    def __init__(self, track):
        self.track = track
        try:
            self.instrument=self.instrument_list[track.channel.instrument]
        except IndexError:
            self.instrument="Unknown"


    def process(self, min_beat_thr, max_beat_thr, rest_thr):
        if self.track.isPercussionTrack or self.instrument == "Unknown":
            return None
        rest_acc = []
        current_chunk = []
        chunks = []
        for measure in self.track.measures:
            for beat in measure.voices[0].beats:
                beat = SimpleBeat(beat)
                if beat.is_chord:
                    # discard tracks with cords
                    return None
                # if the current chunk is too long, add it and stop processing the track
                if len(current_chunk) >= max_beat_thr:
                    chunks.append(current_chunk)
                    return chunks
                if beat.is_rest:
                    rest_acc.append(beat.encoding)

                # the beat has a single note
                elif len(rest_acc) > rest_thr:  # the new note is from a different chunk
                    if len(current_chunk) >= min_beat_thr:  # avoid dumping sequences that are too short
                        chunks.append(np.array(current_chunk, dtype=np.int8))
                    # reset accumulators and parse the current note
                    rest_acc.clear()
                    current_chunk = [beat.encoding]

                elif len(rest_acc):  # new note within the same chunk after a sequence of rests
                    [current_chunk.append(x) for x in rest_acc]  # update accumulator with rest sequence
                    rest_acc.clear()
                    # parse and compute current note
                    current_chunk.append(beat.encoding)

                else:  # new note with no leading rests
                    current_chunk.append(beat.encoding)

        # dump the last notes of the file if the sequence is long enough
        if len(current_chunk) >= min_beat_thr:
            chunks.append(np.array(current_chunk, dtype=np.int8))
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
            min_beat_thr: int,
            max_beat_thr: int
    ):
        self.input_path = in_path
        self.file_paths = get_file_paths(self.input_path)
        self.output_path = out_path
        self.rest_thr = rest_thr
        self.min_beat_thr = min_beat_thr
        self.max_beat_thr = max_beat_thr

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
            chunks = track.process(self.min_beat_thr, self.max_beat_thr, self.rest_thr)
            if chunks:
                self.save_chunks(chunks, track.instrument, song_name, song_index, track_index)

    def save_chunks(
            self,
            chunks: list,
            instrument: str,
            song_name: str,
            song_number: int,
            track_number: int
    ):

        song_name = song_name.lower().lstrip().rstrip()
        inst_folder = os.path.join(self.output_path, "inst_grouped", instrument, f'song {song_number}', f'track {track_number}')
        song_folder = os.path.join(self.output_path, "song_grouped", f'song {song_number}', f'track {track_number}')
        os.makedirs(inst_folder, exist_ok=True)
        os.makedirs(song_folder, exist_ok=True)
        for idx, chunk in enumerate(chunks):
            chunk_name = f"{song_name}({instrument})(track {track_number})(chunk {idx}).npy"
            inst_dest = os.path.join(
                inst_folder,
                chunk_name
            )
            song_dest = os.path.join(
                song_folder,
                chunk_name
            )
            np.save(inst_dest, chunk)
            np.save(song_dest, chunk)
