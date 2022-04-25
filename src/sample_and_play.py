import os
import json
import time
import argparse
import sys
import csv
import math
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
import numpy as np
from src.models.my_model import MyModel
def dir_path(string: str):
    path = os.path.join(
        *(string.split("/"))
    )
    if os.path.exists(path):
        return path
    else:
        raise FileNotFoundError(path)

def expand_config(config):
    config["outputs"] = {
            "semitone": {
                "loss": "wcce",
                "metrics": ["ba", "ac"],
                "loss_weight": 154/1623

            },
            "octave": {
                "loss": "wcce",
                "metrics": ["ba", "ac"],
                "loss_weight": 182/1623

            },
            "dur_log": {
                "loss": "wcce",
                "metrics": ["ba", "ac"],
                "loss_weight": 286/1623

            },
            "dotted": {
                "loss": "wcce",
                "metrics": ["ba", "ac"],
                "loss_weight": 1001/1623

            }
        }
    config["batch_size"] = 1
    return config


# load the configuration of a model, create it, load its weights, predict and Concatenate
def predict():

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
    with open(config_path) as fp:
        run_configs: list = [expand_config(x) for x in json.load(fp)]
    model=MyModel(run_configs[0], output_path, verbose)
    model.load_weights()

    data=model.data.test.map(lambda x,y: x).take(1)
    output = model.predict(data, 20)
    np.save("music.npy",output)

def to_midi():
    from midiutil import MIDIFile
    data=np.load("music.npy")
    st=data[:,0]-1
    oct=data[:,1]-1
    dur_log=(data[:,2]-2).astype(float)
    # dur_log=np.array(range(0,7)).astype(float)-2
    # dur_log=np.power(2,-dur_log)
    dotted=data[:,3]
    dotted[dotted==0]=1
    dotted[dotted==1]=3/2
    track = 0
    channel = 0
    dur=1/np.power(2,dur_log)
    dur=dur*dotted
    tempo = 60  # In BPM
    volume = 100  # 0-127, as per the MIDI standard
    time= 0
    MyMIDI = MIDIFile(1)  # One track, defaults to format 1 (tempo track is created
    # automatically)
    MyMIDI.addTempo(track, time, tempo)

    for i in range(data.shape[0]):
        if st[i]<0 or oct[i]<0:
            time+=dur[i]
            continue
        MyMIDI.addNote(track, channel, oct[i]*12+st[i], time, dur[i], volume)
        time += dur[i]
    with open("major-scale.mid", "wb") as output_file:
        MyMIDI.writeFile(output_file)

to_midi()