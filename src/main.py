import json

import numpy as np
from matplotlib import pyplot as plt

from GP2CSV import gp_to_csv
from src.CSV2Dataset import CSV2Dataset
from src.ModelTrainer import ModelTrainer

if __name__ == "__main__":
    with open("config.json", "r") as fp:
        params = json.load(fp)
    # gp_to_csv(
    #     params['gp_to_csv_input_path'],
    #     params['gp_to_csv_output_path'],
    #     params['max_silence_threshold'],
    #     params['minimum_beats'],
    #     params['gp_test_rate'],
    #     params['track_name']
    # )
    dataset = CSV2Dataset()
    # dataset.create_dataset(
    #     params["dataset_input_path"],
    #     params["dataset_output_path"],
    #     params["input_beats"],
    #     params["label_beats"],
    #     params["save_dataset"]
    # )
    dataset.read_dataset(
        params["csv_to_dataset_output_path"],
        params["input_beats"])
    mt = ModelTrainer(dataset,
                      params["selected_models"],
                      params["batch_size"],
                      params["max_epochs"],
                      params["loss_weights"],
                      params["model_trainer_output_path"],
                      params["create_model"],
                      params["save_model"])
