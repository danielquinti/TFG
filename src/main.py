
import numpy as np

from src.CSV2Dataset import CSV2Dataset
from src.ModelTrainer import ModelTrainer

if __name__ == "__main__":
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