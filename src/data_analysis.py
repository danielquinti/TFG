import glob
import numpy as np
import itertools
import json
import os

from data_processing.dataset_manager import *
from model_trainer import ModelTrainer
from sklearn.metrics import confusion_matrix
#
#
# cm_tags = {
#     "notes": ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B", "REST"],
#     "duration": ["1", "1/2", "1/4", "1/8", "1/16", "1/32", "1/64", "1/128"]
# }
# #
# #
# # def plot_confusion_matrix(cm, classes,
# #                           normalize=False,
# #                           title='Confusion matrix',
# #                           cmap=plt.cm.Blues):
# #     """
# #     This function prints and plots the confusion matrix.
# #     Normalization can be applied by setting `normalize=True`.
# #     """
# #     plt.imshow(cm, interpolation='nearest', cmap=cmap)
# #     plt.title(title)
# #     plt.colorbar()
# #     tick_marks = np.arange(len(classes))
# #     plt.xticks(tick_marks, classes, rotation=45)
# #     plt.yticks(tick_marks, classes)
# #
# #     if normalize:
# #         cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
# #
# #     thresh = cm.max() / 2.
# #     for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
# #         plt.text(j, i, cm[i, j],
# #                  horizontalalignment="center",
# #                  color="white" if cm[i, j] > thresh else "black")
# #
# #     plt.ylabel('True label')
# #     plt.xlabel('Predicted label')
# #     plt.show()
# #
# #
# def get_metrics():
#     os.chdir(
#         os.path.join(
#             "data",
#             "cm"
#         )
#     )
#     # plt.rcParams['figure.figsize'] = [15, 8]
#     for name in glob.glob("*.csv"):
#         data = np.loadtxt(name, dtype=int)
#         mean_ap = np.average([data[i, i] / np.sum(data[i, :]) for i in range(data.shape[0])])
#         accuracy = np.sum([data[i, i] for i in range(data.shape[0])]) / np.sum(data)
#         print(name)
#         print("    Accuracy %.2f" % accuracy)
#         print("    MAP %.3f" % mean_ap)
#         # tags = cm_tags["notes"] if "notes" in name else cm_tags["duration"]
#         display_name = name.split(".")[0]
#         # plot_confusion_matrix(data,
#         #                       tags,
#         #                       title=display_name,
#         #                       cmap='Blues',
#         #                       # display_labels=data.target_names
#         #                       )
#     os.chdir(
#         os.path.join(
#             "..",
#             ".. "
#         )
#     )
def save_cms():
    dm = DatasetManager()
    dataset = dm.load_dataset()
    mt = ModelTrainer(dataset)
    mt.load_models()
    for name, model in mt.trained_models.items():

        notes_pred, duration_pred = model.predict(
                dataset.test.inputs
            )
        notes_pred = np.argmax(notes_pred, axis=1)
        notes_true=np.argmax(dataset.test.labels.notes, axis=1)
        data = confusion_matrix(
            notes_true,
            notes_pred
        )
        mean_ap = (np.diag(data) / np.sum(data, axis=1)).mean()
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

        mean_ap = (np.diag(data) / (np.sum(data, axis=1)+1e-8)).mean()
        accuracy = np.sum(np.diag(data)) / np.sum(data)
        print(name+" duration")
        print("    Accuracy %.2f" % accuracy)
        print("    MAP %.3f" % mean_ap)

save_cms()
