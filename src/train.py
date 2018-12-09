import os

import config as cfg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.callbacks import Callback
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten
from keras.layers import Dropout
from keras.layers import regularizers
from keras.models import Sequential
from keras.models import load_model
from keras.optimizers import SGD
from skimage.io import imread
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error
from tensorflow import set_random_seed

#Reads train and validation dataSet
train_data = pd.read_csv("train2.csv")
valid_data = pd.read_csv("valid2.csv")

INPUT_PATH = cfg.DATASET_PATH + "/"
TRAIN_RATE = cfg.TRAIN_RATE
VALID_RATE = cfg.VALID_RATE
TEST_RATE = cfg.TEST_RATE

# Calculates the set sizes
train_size = len(train_data)
valid_size = len(valid_data)

# Initializes the set inputs and outputs
train_samples = np.zeros(train_size, dtype=[('input', float, (256,256)), ('output', float, 2)])
valid_samples = np.zeros(valid_size, dtype=[('input', float, (256,256)), ('output', float, 2)])

#Gets training and validation images as input, and gets their output
for index, row in train_data.iterrows():
    train_samples[index] = imread(INPUT_PATH + "train/" + row['Image_Index'])/255, row[7:]

for index, row in valid_data.iterrows():
    valid_samples[index] = imread(INPUT_PATH + "valid/" + row['Image_Index'])/255, row[7:]

# Resizes the inputs of the datasets
train_x = np.reshape(train_samples['input'], (len(train_samples), 256, 256, 1))
valid_x = np.reshape(valid_samples['input'], (len(valid_samples), 256, 256, 1))

# Size of the output layer
nb_classes = 2

# Generate random seeds
np.random.seed(1)
set_random_seed(2)


class TrainingHistory(Callback):

    # Initializes empty lists for metric storing
    def on_train_begin(self, logs={}):
        # Error on training data
        self.losses = []
        # Error on validation data
        self.valid_losses = []
        # Stores how good is the model (on training data)
        self.accs = []
        # Stores how good is the model (on validation data)
        self.valid_accs = []
        # Number of epochs
        self.epoch = 0

    # At the end of an epoch, save the performance of the actual network
    def on_epoch_end(self, epoch, logs={}):
        self.losses.append(logs.get('loss'))
        self.valid_losses.append(logs.get('val_loss'))
        self.accs.append(logs.get('acc'))
        self.valid_accs.append(logs.get('val_acc'))
        self.epoch += 1



# Implements the network
history = TrainingHistory()

model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(5, 5), strides= 1, padding='same', activation='tanh', input_shape=(256, 256, 1)))
model.add(Conv2D(filters=64, kernel_size=(5, 5), strides= 1, padding='same', activation='tanh'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

model.add(Conv2D(filters=128, kernel_size=(5, 5), strides=1, padding='same', activation='tanh'))
model.add(Conv2D(filters=128, kernel_size=(5, 5), strides= 1, padding='same', activation='tanh'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=0.0001, nesterov=True), metrics=['accuracy'])

# Configure early stopping
from keras.callbacks import EarlyStopping
patience = 20
early_stopping = EarlyStopping(patience=patience, verbose=1)

# Saves the best model (using validation error)
from keras.callbacks import ModelCheckpoint
checkpointer = ModelCheckpoint(filepath='weights-11.hdf5', save_best_only=True, verbose=1)

# Reduces learning rate automatically
from keras.callbacks import ReduceLROnPlateau
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=5, min_lr=10e-5)

# Starts training
model.fit(train_x, train_samples['output'],
          # Size of our batch
          batch_size=10,
          # Number of epochs
          nb_epoch=200,
          # Verbose parameter
          verbose=1,
          # Validation runs in parallel with training
          validation_data=(valid_x, valid_samples['output']),
          # Save important metrics in 'history'
          callbacks=[early_stopping, reduce_lr, checkpointer, history], # Shuffle input data
          shuffle = True)

# Plots and save a diagram about the Measure of error
plt.figure(figsize=(10, 5))
plt.title('Measure of error')
plt.plot(np.arange(history.epoch), history.losses, color ='g', label='Measure of error on training data')
plt.plot(np.arange(history.epoch), history.valid_losses, color ='b', label='Measure of error on validation data')
plt.legend(loc='upper right')
plt.xlabel('Number of training iterations')
plt.ylabel('y')
plt.grid(True)
plt.savefig("12-09.png")

# Plots and save a diagram about the Measure of accuracy
plt.figure(figsize=(10, 5))
plt.title('Measure of accuracy')
plt.plot(np.arange(history.epoch), history.accs, color ='g', label='Measure of accuracy on training data')
plt.plot(np.arange(history.epoch), history.valid_accs, color ='b', label='Measure of accuracy on validation data')
plt.legend(loc='upper right')
plt.xlabel('Number of training iterations')
plt.ylabel('y')
plt.grid(True)
plt.savefig("12-09acc.png")
