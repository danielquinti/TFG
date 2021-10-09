import json
from configparser import ConfigParser
from GP2CSV import gp_to_csv
from src.CSV2Dataset import CSV2Dataset
from src.ModelTrainer import ModelTrainer

if __name__ == "__main__":
    with open("config.json", "r") as fp:
        params = json.load(fp)
    gp_to_csv(
        params['gp_to_csv_input_path'],
        params['gp_to_csv_output_path'],
        params['max_silence_threshold'],
        params['minimum_beats'],
        params['gp_test_rate'],
        params['track_name']
    )
    dataset = CSV2Dataset(create=False)
    # train_note = dataset.train_labels["repeated_note"]
    # train_dur = dataset.train_labels["repeated_duration"]
    # test_note = dataset.test_labels["repeated_note"]
    # test_dur = dataset.test_labels["repeated_duration"]
    # print(f"Train : {np.sum(train_note)/train_note.shape[0]*100}% repeated notes")
    # print(f"Train : {np.sum(train_dur)/train_dur.shape[0]*100}% repeated duration")
    # print(f"Test : {np.sum(test_note)/test_note.shape[0]*100}% repeated notes")
    # print(f"Test : {np.sum(test_dur)/test_dur.shape[0]*100} % repeated duration")
    benchmark = ModelTrainer(dataset)
    # benchmark.execute()