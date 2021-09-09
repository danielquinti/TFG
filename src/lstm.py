#!/usr/bin/python3

import os
import numpy as np
os.environ["TF_CPP_MIN_LOG_LEVEL"]="2"

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

N=6
input_size=12
output_size=12
seq_length=N-1
batch_size=64
epochs=5
os.chdir("data")
train=np.loadtxt("train.csv",delimiter=",")
test=np.loadtxt("test.csv",delimiter=",")
train_mask = np.ones(train.shape[0], dtype=bool)
test_mask = np.ones(test.shape[0], dtype=bool)
train_mask[6::6] = 0
test_mask[6::6] = 0
x_train=train[train_mask,:]
x_test=test[test_mask,:]
train_mask=~np.array(train_mask)
test_mask=~np.array(test_mask)
y_train=train[train_mask,:]
y_test=test[test_mask,:]

model=keras.models.Sequential()
model.add(keras.Input(shape=(input_size,seq_length)))
model.add(layers.SimpleRNN(128,activation='relu'))
model.add(layers.Dense(output_size))
print(model.summary())

loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True)
optim=keras.optimizers.Adam(lr=0.001)
metrics=["accuracy"]
model.compile(loss=loss,optimizer=optim,metrics=metrics)
model.fit(x_train,y_train,batch_size=batch_size,epochs=epochs,verbose=2)
model.evaluate(x_test,y_test,batch_size=batch_size,verbose=2)