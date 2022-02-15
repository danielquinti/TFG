import argparse
import os
import sys

sys.path.append(
    os.path.join(
        "src",
        "architecture"
    )
)
sys.path.append(
    os.path.join(
        "src",
        "data_processing"
    )
)
import model_trainer


def dir_path(string: str):
    path = os.path.join(
        *(string.split("/"))
    )
    if os.path.exists(path):
        return path
    else:
        raise FileNotFoundError(path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=dir_path)
    parser.add_argument(
        '--output_path',
        default="results"
    )
    parser.add_argument(
        '--verbose',
        default=1,
    )
    args = parser.parse_args()
    config_path = args.path
    output_path = args.output_path
    verbose = args.verbose
    mt = model_trainer.ModelTrainer(
        config_path,
        output_path,
        verbose
    )
    mt.run_all()
