import os
import json
import time
import argparse
import sys
import csv
from collections import defaultdict

from src.song_processing.song_processor import get_file_paths, open_song

sys.path.append("src")
from pipeline import pipeline
from song_processing import song_processor as sp


def dir_path(string: str):
    path = os.path.join(
        *(string.split("/"))
    )
    if os.path.exists(path):
        return path
    else:
        raise FileNotFoundError(path)


def encode():
    if __name__ == "__main__":
        start_time = time.time()
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
        silence_thr: int = params["silence_thr"]
        min_beats: int = params["min_beats"]
        max_beats: int = params["max_beats"]
        parser = sp.SongProcessor(
            input_path,
            output_path,
            silence_thr,
            min_beats,
            max_beats
        )
        parser.process_songs()
        print("--- %s seconds ---" % (time.time() - start_time))


def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=dir_path)
    parser.add_argument(
        '--output_path',
        default="report"
    )
    parser.add_argument(
        '--verbose',
        default=1,
    )
    args = parser.parse_args()
    config_path = args.path
    output_path = args.output_path
    verbose = args.verbose
    config_name = os.path.splitext(os.path.basename(config_path))[0]
    reports = pipeline.Pipeline(
        config_path,
        output_path,
        verbose
    ).run()
    with open(
            os.path.join(
                output_path,
                f'{config_name}_metrics_report.csv'
            ),
            'w',
    ) as csv_file:
        writer = csv.writer(csv_file)
        for report in reports:
            writer.writerow(report[0])
            writer.writerow(report[1])

def test():
    from tensorflow.keras import layers
    import tensorflow as tf
    import numpy as np
    input_shape=(30,4)
    number_of_classes = {
        # "octave" :11,
        "semitone" :13,
    }
    window_size=31
    input_beats=30
    batch_size=32
    def out_prep():
        x=tf.keras.layers.Input(shape=4)
        y=x[:,2]
        y=tf.cast(y,tf.int32)
        y=tf.one_hot(y,depth=number_of_classes["semitone"])
        return tf.keras.Model(inputs=x,outputs=y)


    inputs = layers.Input(shape=input_shape, dtype=tf.uint8)
    window_beats = tf.unstack(inputs, axis=1)
    raw_outputs = tf.unstack(window_beats[-1], axis=-1)
    oh_outputs = [tf.one_hot(raw_output, depth=n_classes) for raw_output, n_classes in zip(raw_outputs, number_of_classes.values())]
    outputs = [
        layers.Layer(
            trainable=False,
            name=name
        )(data) for name, data in zip(number_of_classes.keys(), oh_outputs)]
    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    folder_path = os.path.join("data","modest", "train", str(window_size))
    data_path = os.path.join(folder_path, "windows.npy")
    data = np.load(data_path).reshape((-1, window_size, 4))
    ds = tf.data.Dataset.from_tensor_slices(
        (
            data[:, :input_beats, :],
            data[:, input_beats, :]
        )
    )
    ds = ds.shuffle(buffer_size=1000)
    ds = ds.batch(32, drop_remainder=True)
    ds = ds.map(
        lambda x, y: (x, out_prep()(y)),
        num_parallel_calls=tf.data.AUTOTUNE
    ).prefetch(tf.data.AUTOTUNE)
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    model.fit(ds, epochs=1)

def raw_insight():
    # read all files from dirty tabs, get how many tracks per instrument
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
        "Pad 1 (a warm pad stacked with a bell)",
        "Pad 2 (warm)",
        "Pad 3 (polysynth)",
        "Pad 4 (choir)",
        "Pad 5 (bowed glass or bowed)",
        "Pad 6 (metallic)",
        "Pad 7 (halo)",
        "Pad 8 (sweep)",
        "FX 1 (rain)",
        "FX 2 (a bright perfect fifth pad)",
        "FX 3 (crystal)",
        "FX 4 (atmosphere, nylon-like sound)",
        "FX 5 (brightness)",
        "FX 6 (goblins)",
        "FX 7 (echoes)",
        "FX 8 (sci-fi theme)",
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
    inst_dict=dict.fromkeys(instrument_list,0)
    track_counter=0
    song_counter=0
    in_path=os.path.join("data","dirty_tabs")
    file_names =get_file_paths(in_path)
    for song_path in file_names:
        song = open_song(song_path)
        if song is None:
            continue
        for track in song.tracks:
            inst_dict[instrument_list[track.channel.instrument]]+=1
            track_counter+=1
            print(f'{song_counter} songs analyzed and {track_counter} tracks')
        song_counter+=1
    with open('inst.json', 'w') as fp:
        json.dump(inst_dict, fp)
    print(len(file_names))

if __name__ == "__main__":
    raw_insight()