import numpy as np
import matplotlib.pyplot as plt
import cv2
import config as cfg
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten
from keras.optimizers import SGD
from sklearn import preprocessing
import os
from keras.callbacks import Callback
from tensorflow import set_random_seed
from keras.models import load_model
from sklearn.metrics import mean_squared_error


# Load the already processed csv file
entry = pd.read_csv("data_first_folder.csv")

# Calculate the set sizes
train_size = cfg.TRAIN_RATE * len(entry)
valid_size = cfg.VALID_RATE * len(entry)
test_size = cfg.TEST_RATE * len(entry)

# Initialize the set inputs and outputs
samples = np.zeros(len(entry), dtype=[('input', float, (256,256)), ('output', float, 15)])


# Reads an image form the dataset, converts it to grayscale, resizes it and scales it
def readandscale(im_row):
    # Reads out the name of the picture (Image_Index) and reads the image in grayscale
    src = cv2.imread(cfg.EXT_DATASET_PATH + "/" + im_row['Image_Index'], cv2.IMREAD_GRAYSCALE)
    # Resize the image
    resized = cv2.resize(src, (cfg.IMG_SIZE, cfg.IMG_SIZE))
    # Scale the image
    return preprocessing.minmax_scale(np.array(resized, dtype='float64'))

# Iterates through the rows of the dataframe
# Rows in the csv file are already mixed
for index, row in entry.iterrows():
    # First, it fills the trainingset, than the validation and the testset
    # If the image exists in the dataset, we scale and resize it, then we add it to the set's inputs
    # Outputs are always in the Finding_Labels column
    print(index)
    if os.path.isfile(cfg.EXT_DATASET_PATH + "/" + row['Image_Index']):
        img_scaled = readandscale(row)
        samples[index] = img_scaled, row[7:]

# Makes training, validation and test datasets
train = samples[0:int(len(entry)*(1-cfg.VALID_RATE-cfg.TEST_RATE))]
valid = samples[int(len(entry)*(1-cfg.VALID_RATE-cfg.TEST_RATE)):int(len(entry)*(1-cfg.TEST_RATE))]
test = samples[int(len(entry)*(1-cfg.TEST_RATE)):]

# Separates the inputs of the datasets
train_x = np.reshape(train['input'], (len(train), 256, 256, 1))
valid_x = np.reshape(valid['input'], (len(valid), 256, 256, 1))
test_x = np.reshape(test['input'], (len(test), 256, 256, 1))
print(train[2])

 #-------------------Model--------------------------

# Size of the output layer
nb_classes = 15

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
model.add(Conv2D(filters=32, kernel_size=(5, 5), strides= 1, padding='same', activation='tanh',
                 input_shape=(256, 256, 1)))
model.add(Conv2D(filters=64, kernel_size=(5, 5), strides= 1, padding='same', activation='tanh'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Conv2D(filters=128, kernel_size=(5, 5), strides=1, padding='same', activation='tanh'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='tanh'))
model.add(Dense(nb_classes, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=0.001), metrics=['accuracy'])


# Configure early stopping
from keras.callbacks import EarlyStopping
patience = 10
early_stopping = EarlyStopping(patience=patience, verbose=1)

# Saves the best model (using validation error)
from keras.callbacks import ModelCheckpoint
checkpointer = ModelCheckpoint(filepath='weights.hdf5', save_best_only=True, verbose=1)

# Reduces learning rate automatically
from keras.callbacks import ReduceLROnPlateau
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=20, min_lr=10e-5)


model.fit(train_x, train['output'],
          # Size of our batch
          batch_size=20,
          # Number of epochs
          nb_epoch=100,
          # Verbose parameter
          verbose=1,
          # Validation runs in parallel with training
          validation_data=(valid_x, valid['output']),
          # Save important metrics in 'history'
          callbacks=[reduce_lr, checkpointer, early_stopping, history],
          # Shuffle input data
          shuffle=True)


plt.figure(figsize=(10, 5))
plt.title('Measure of error')
plt.plot(np.arange(history.epoch), history.losses, color ='g', label='Measure of error on training data')
plt.plot(np.arange(history.epoch), history.valid_losses, color ='b', label='sure of error on validation data')
plt.legend(loc='upper right')
plt.xlabel('Number of training iterations')
plt.ylabel('y')
plt.grid(True)
plt.show()

# Load the best model
model = load_model('weights.hdf5')

# Predicating with test data
preds = model.predict(test_x)

# Calculating the error on test data
test_mse = mean_squared_error(test['output'], preds)
print("Test MSE: %f" % (test_mse))
model.summary()
