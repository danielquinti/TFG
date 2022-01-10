import os
import json
import model_trainer
from data_processing import song_splitter

if __name__ == "__main__":
    # song_splitter.split_songs()
    mt = model_trainer.ModelTrainer()
    mt.run()
