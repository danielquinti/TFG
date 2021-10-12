import math
from src.data_processing.utils import *
from src.data_processing.Dataset import *


def get_distribution(input_path,
                     input_beats,
                     window_beats,
                     distribution_name):
    file_names = get_file_paths(os.path.join(input_path, distribution_name))
    inputs = []
    label_notes = []
    label_duration = []
    for file_name in file_names:
        contents = np.loadtxt(file_name)
        # add inputs and labels by sliding window
        for i in range(contents.shape[0] - window_beats):
            inputs.append(contents[i:i + input_beats])
            label_beat = contents[i + input_beats]

            duration = np.zeros(8)
            duration[-int(round(math.log2(np.max(label_beat))))] = 1

            label_notes.append(np.sign(label_beat))
            label_duration.append(duration)

    inputs = np.array(inputs)
    labels = Labels(np.array(label_notes), np.array(label_duration))
    return Distribution(inputs, labels)


def csv_to_dataset(
        input_path,
        input_beats,
        label_beats):
    window_beats = input_beats + label_beats
    train = get_distribution(input_path,
                             input_beats,
                             window_beats,
                             "train")
    test = get_distribution(input_path,
                            input_beats,
                            window_beats,
                            "test")
    return Dataset(train, test)
