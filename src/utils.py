import os


def get_file_names(route):
    name_list = []
    for root, dirs, files in os.walk(route):
        for file in files:
            name_list.append(os.path.join(root, file))
    return name_list