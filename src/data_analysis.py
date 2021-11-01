import glob
import numpy as np
import itertools
import json
import os

from data_processing.dataset_manager import *
from model_trainer import ModelTrainer
from sklearn.metrics import confusion_matrix

from model.metrics import BalancedAccuracy


def compute_metrics(raw_true, raw_pred, name, nd, dist):
    true = np.argmax(raw_true, axis=1)
    pred = np.argmax(raw_pred, axis=1)
    data = confusion_matrix(
        true,
        pred
    )
    diag = np.diag(data)
    true_counts = np.sum(data, axis=1)

    # avoid division by 0
    recalls = np.divide(diag, true_counts, out=np.zeros_like(diag, dtype='float'), where=true_counts != 0)
    # compute mean discarding labels with no counts
    bal_acc = recalls.sum() / np.sign(true_counts).sum()

    print(name + nd+dist)
    print("    Balanced Accuracy %.3f" % bal_acc)
    output_dir=os.path.join("data", "cm")
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    np.savetxt(
        os.path.join(
            output_dir,
            f'{name}_{nd}_{dist}.csv'
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

        compute_metrics(dataset.test.labels.notes,notes_pred, name, "notes","test")
        compute_metrics(dataset.test.labels.duration, duration_pred, name, "duration", "test")
        notes_pred, duration_pred = model.predict(
            dataset.train.inputs
        )

        compute_metrics(dataset.train.labels.notes,notes_pred, name, "notes", "train")
        compute_metrics(dataset.train.labels.duration, duration_pred, name, "duration", "train")

def compute_metrics_test(raw_true,raw_pred):
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
    print("    Balanced Accuracy %.3f" % bal_acc)

def test():
    m=BalancedAccuracy(3)
    true=[
        [1.,0.,0.],
        [1.,0.,0.],
        [1.,0.,0.],
        [1.,0.,0.],
        [1.,0.,0.],
        [0.,1.,0.]
    ]

    pred=[
        [0., 1., 0.],
        [0., 1., 0.],
        [0., 1., 0.],
        [0., 1., 0.],
        [0., 1., 0.],
        [0., 1., 0.]
    ]
    m.update_state(true,pred)
    print(float(m.result()))
    compute_metrics(true,pred)

save_cms()