import json

from keras.models import load_model

from data_processing.GP2CSV import gp_to_csv
from data_processing.CSV2Dataset import *
from src.ModelTrainer import ModelTrainer

if __name__ == "__main__":
    with open("src\\config.json", "r") as fp:
        params = json.load(fp)
    # gp_to_csv(
    #     params['gp_to_csv_input_path'],
    #     params['gp_to_csv_output_path'],
    #     params['max_silence_threshold'],
    #     params['minimum_beats'],
    #     params['gp_test_rate'],
    #     params['track_name']
    # )
    # dataset = csv_to_dataset(
    #     params["csv_to_dataset_input_paths"],
    #     params["input_beats"],
    #     params["label_beats"],
    # )
    # dataset.save(params["csv_to_dataset_output_path"])
    dataset = load_dataset(
        params["csv_to_dataset_output_path"],
        params["input_beats"]
    )
    mt = ModelTrainer()
    mt.train_models(dataset,
                    params["selected_models"],
                    params["batch_size"],
                    params["max_epochs"],
                    params["loss_weights"],
                    params["model_trainer_output_path"],
                    params["save_model"])

    # os.chdir(params["model_trainer_output_path"])
    # model = load_model("ffwd")
    # a = model.predict(dataset.test.inputs)
    # print(a)
