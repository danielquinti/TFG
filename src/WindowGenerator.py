import os
import math
import numpy as np
import random


class WindowGenerator():
    def __init__(self, input_width,label_width,test_rate):
        self.input_width=input_width
        self.label_width=label_width
        self.window_width=input_width+label_width
        self.test_rate=test_rate
        self.inputs=[]
        self.labels=[]
        self.train_inputs=[]
        self.test_inputs=[]
        self.train_labels={"notes":[],"duration":[]}
        self.test_labels={"notes":[],"duration":[]}
        self.generate_window()
        self.split()

    def get_file_list(self, route):
        file_list = []
        for root, dirs, files in os.walk(route):
            for file in files:
                file_list.append(os.path.join(root, file))
        return file_list

    def crawl_file(self, file_name):
        contents=np.loadtxt(file_name)
        contents=contents[:(contents.shape[0]//self.window_width)*self.window_width]
        [self.inputs.append([contents[i+self.window_width*j] for i in range(self.input_width)]) for j in range(contents.shape[0]//self.window_width)]
        # [self.labels.append([contents[i+self.window_width*j-2] for i in range(self.label_width)]) for j in range(1,math.ceil(contents.shape[0]/self.window_width)+1)]
        for j in range(1, math.ceil(contents.shape[0] / self.window_width) + 1):
            for i in range(self.label_width):
                self.labels.append(Label(np.array(contents[i+self.window_width*j-self.label_width])))
        # for j in range(1, math.ceil(contents.shape[0] / self.window_width) + 1):
        #     for i in range(self.label_width):
        #         print(i+self.window_width*j-self.label_width)

    def generate_window(self):
        file_names=self.get_file_list("data\\dump1")
        for file_name in file_names:
            self.crawl_file(file_name)
        self.inputs=random.sample(self.inputs,len(self.inputs))
        self.labels=random.sample(self.labels,len(self.labels))

    def split(self):
        self.train_inputs=self.inputs[:math.floor(((1-self.test_rate)*len(self.inputs)))]
        self.test_inputs=self.inputs[math.ceil(((1-self.test_rate)*len(self.inputs))):]
        self.train_inputs=np.array(self.train_inputs)
        self.test_inputs=np.array(self.test_inputs)

        aux=self.labels[:math.floor(((1-self.test_rate)*len(self.labels)))]
        self.train_labels["notes"]=np.array([[label.notes] for label in aux])
        self.train_labels["duration"]=np.array([label.duration for label in aux])

        aux=self.labels[math.ceil(((1-self.test_rate)*len(self.labels))):]
        self.test_labels["notes"]=np.array([[label.notes] for label in aux])
        self.test_labels["duration"]=np.array([label.duration for label in aux])

class Label():
    def __init__(self,beat):
        self.notes=beat
        self.duration=np.max(beat)