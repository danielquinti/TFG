import json

from keras.models import load_model

from data_processing.GP2CSV import gp_to_csv
from data_processing.CSV2Dataset import *
from ModelTrainer import ModelTrainer

if __name__ == "__main__":
    with open(os.path.join("src", "config.json"), "r") as fp:
        params = json.load(fp)
    # gp_to_csv(
    #     os.path.join(
    #         params["gp_to_csv_input_path"][0],
    #         params["gp_to_csv_input_path"][1],
    #     ),
    #     os.path.join(
    #         params["gp_to_csv_output_path"][0],
    #         params["gp_to_csv_output_path"][1],
    #     ),
    #     params['gp_to_csv_output_path'],
    #     params['max_silence_threshold'],
    #     params['minimum_beats'],
    #     params['gp_test_rate'],
    #     params['track_name']
    # )
    dataset = csv_to_dataset(
        os.path.join(
            params["csv_to_dataset_input_path"][0],
            params["csv_to_dataset_input_path"][1],
        ),
        params["input_beats"],
        params["label_beats"],
    )
    dataset.save(
        os.path.join(
            params["csv_to_dataset_output_path"][0],
            params["csv_to_dataset_output_path"][1],
        )
    )
    # dataset = load_dataset(
    #     os.path.join(
    #         params["csv_to_dataset_output_path"][0],
    #         params["csv_to_dataset_output_path"][1],
    #     ),
    #     params["input_beats"]
    # )
    mt = ModelTrainer()
    mt.train_models(dataset,
                    params["selected_models"],
                    params["batch_size"],
                    params["max_epochs"],
                    params["loss_weights"],
                    params["model_trainer_output_path"],
                    params["save_model"]
    )

    # os.chdir(params["model_trainer_output_path"])
    # model = load_model("ffwd")
    # a = model.predict(dataset.test.inputs)
    # print(a)
