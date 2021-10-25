import json

from keras.models import load_model

from data_processing.gp_to_csv import gp_to_csv
from data_processing.dataset_manager import *
from model_trainer import ModelTrainer
if __name__ == "__main__":
    with open(os.path.join("src", "train_config.json"), "r") as fp:
        params = json.load(fp)

    # gp_to_csv()
    for i in (5,10,15,20):
        dm = DatasetManager(i)
        dataset = dm.extract_dataset()
        # dm.save_dataset()
        mt= ModelTrainer(dataset)
        mt.train_models()