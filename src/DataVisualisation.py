import itertools

from matplotlib import pyplot as plt
import numpy as np
from utils import get_file_names

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')

cm_tags = {
    "notes": ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B", "REST"],
    "duration": ["1", "1/2", "1/4", "1/8", "1/16", "1/32", "1/64", "1/128"]
}

def graph_trainval_metrics(config):
    x=range(config.max_epochs)
    for name in get_file_names(config.output_dir+"\\trainval\\"):
        y=np.loadtxt(name).transpose()
        plt.plot(x, y, label=name.split("\\")[-1].split(".")[0])
    plt.legend()
    plt.show()


def plot_confusion_matrices(config):
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(15, 10))
    for filename, ax in zip(get_file_names(config.output_dir+"\\cm\\"), axes.flatten()):
        plot_confusion_matrix(np.loadtxt(filename,dtype=int),
                              cm_tags["notes"],
                              cmap='Blues',
                              #display_labels=data.target_names
                              )
        #ax.title.set_text(type(cls).__name__)
    plt.tight_layout()
    plt.show()

# fig, axs = plt.subplots(2, 2)
# axs[0, 0].plot(x, y)
# axs[0, 0].set_title('Axis [0, 0]')
# axs[0, 1].plot(x, y, 'tab:orange')
# axs[0, 1].set_title('Axis [0, 1]')
# axs[1, 0].plot(x, -y, 'tab:green')
# axs[1, 0].set_title('Axis [1, 0]')
# axs[1, 1].plot(x, -y, 'tab:red')
# axs[1, 1].set_title('Axis [1, 1]')