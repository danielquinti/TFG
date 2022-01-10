import os
import json
import model_trainer
from data_processing import song_splitter

if __name__ == "__main__":
    with open(os.path.join("src", "config", "train_config.json"), "r") as fp:
        params = json.load(fp)
    song_splitter.split_songs()
    # mt = model_trainer.ModelTrainer()
    # mt.run()
