import glob
import numpy as np
import itertools
import json
import os

from data_processing.dataset_manager import *
from model_trainer import ModelTrainer
from sklearn.metrics import confusion_matrix

def save_cms():
    mt = ModelTrainer()
    mt.load_models()
    for name, (model,dataset) in mt.trained_models.items():

        notes_pred, duration_pred = model.predict(
                dataset.test.inputs
            )
        notes_pred = np.argmax(notes_pred, axis=1)
        notes_true=np.argmax(dataset.test.labels.notes, axis=1)
        data = confusion_matrix(
            notes_true,
            notes_pred
        )
        mean_ap = (np.diag(data) / (np.sum(data, axis=1) + 1e-8))
        mean_ap = mean_ap.sum() / np.sign(mean_ap).sum()
        accuracy = np.sum(np.diag(data)) / np.sum(data)
        print(name+" notes")
        print("    Accuracy %.2f" % accuracy)
        print("    MAP %.3f" % mean_ap)
        np.savetxt(
            os.path.join(
                "data",
                "cm",
                f'{name}_notes.csv'
            ),
            data,
            fmt='%i'
        )

        duration_pred, duration_pred = model.predict(
                dataset.test.inputs
            )
        duration_pred = np.argmax(duration_pred, axis=1)
        duration_true=np.argmax(dataset.test.labels.duration, axis=1)
        data = confusion_matrix(
            duration_true,
            duration_pred
        )
        np.savetxt(
            os.path.join(
                "data",
                "cm",
                f'{name}_duration.csv'
            ),
            data,
            fmt='%i'
        )

        mean_ap = (np.diag(data) / (np.sum(data, axis=1) + 1e-8))
        mean_ap = mean_ap.sum() / np.sign(mean_ap).sum()
        accuracy = np.sum(np.diag(data)) / np.sum(data)
        print(name+" duration")
        print("    Accuracy %.2f" % accuracy)
        print("    MAP %.3f" % mean_ap)

save_cms()
