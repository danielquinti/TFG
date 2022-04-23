import os
import json
import time
import argparse
import sys
import csv
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


if __name__ == "__main__":
    train()
