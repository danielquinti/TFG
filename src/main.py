import json

from keras.models import load_model

from data_processing.gp_to_csv import gp_to_csv
from data_processing.dataset_manager import *
from model_trainer import ModelTrainer
from matplotlib import pyplot as plt

if __name__ == "__main__":
    with open(os.path.join("src", "train_config.json"), "r") as fp:
        params = json.load(fp)

    # gp_to_csv()

    dm = DatasetManager()
    dataset = dm.extract_dataset()
    dm.save_dataset()

    # freqs = np.sum(dataset.train.labels.notes, axis=0)
    # plt.bar(range(len(freqs)), freqs, width=0.8)
    # plt.show()
    #
    # freqs = np.sum(dataset.train.labels.duration, axis=0)
    # plt.bar(range(len(freqs)), freqs, width=0.8)
    # plt.show()



    mt = ModelTrainer(dataset)
    mt.run()

