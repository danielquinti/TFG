import json

from keras.models import load_model

from data_processing.gp_to_csv import gp_to_csv
from data_processing.dataset_manager import *
from model_trainer import ModelTrainer
if __name__ == "__main__":
    with open(os.path.join("src", "config", "train_config.json"), "r") as fp:
        params = json.load(fp)

    mt = ModelTrainer()
    mt.run()