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
    parser.add_argument('--path', type=dir_path)
    args = parser.parse_args()
    config_path = args.path
    mt = model_trainer.ModelTrainer(config_path)
    mt.run_all()
