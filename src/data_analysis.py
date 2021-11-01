import glob
import numpy as np
import itertools
import json
import os

from data_processing.dataset_manager import *
from model_trainer import ModelTrainer
from sklearn.metrics import confusion_matrix


def compute_metrics(raw_pred, raw_true, name, nd):
    pred = np.argmax(raw_pred, axis=1)
    true = np.argmax(raw_true, axis=1)
    data = confusion_matrix(
        true,
        pred
    )
    diag = np.diag(data)
    true_counts = np.sum(data, axis=1)

    # avoid division by 0
    recalls = np.divide(diag,true_counts,out=np.zeros_like(diag,dtype='float'),where= true_counts != 0)

    bal_acc = recalls.sum() / np.sign(true_counts).sum()
    print(name + nd)
    print("    Balanced Accuracy %.3f" % bal_acc)
    output_dir=os.path.join("data", "cm")
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    np.savetxt(
        os.path.join(
            output_dir,
            f'{name}_{nd}.csv'
        ),
        data,
        fmt='%i'
    )


def save_cms():
    mt = ModelTrainer()
    mt.load_models()
    for name, (model, dataset) in mt.trained_models.items():
        notes_pred, duration_pred = model.predict(
            dataset.test.inputs
        )

        #compute_metrics(duration_pred, dataset.test.labels.duration, name, "duration")
        compute_metrics(notes_pred, dataset.test.labels.notes, name, "notes")


save_cms()
