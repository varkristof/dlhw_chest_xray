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
 # Load the already processed csv file
entry = pd.read_csv("data_first_folder.csv")
 # Calculate the set sizes
train_size = cfg.TRAIN_RATE * len(entry)
valid_size = cfg.VALID_RATE * len(entry)
test_size = cfg.TEST_RATE * len(entry)
 # Initialize the set inputs and outputs
training_IN = []
valid_IN = []
test_IN = []
training_OUT = []
valid_OUT = []
test_OUT = []
 # Reads an image form the dataset, converts it to grayscale, resizes it and scales it
def readandscale(im_row):
     # Reads out the name of the picture (Image_Index) and reads the image in grayscale
    src = cv2.imread(cfg.EXT_DATASET_PATH + "\\" + im_row['Image_Index'], cv2.IMREAD_GRAYSCALE)
     # Resize the image
    resized = cv2.resize(src, (cfg.IMG_SIZE, cfg.IMG_SIZE))
     # Scale the image
    return preprocessing.minmax_scale(resized)
 # Iterates through the rows of the dataframe
# Rows in the csv file are already mixed
for index, row in entry.iterrows():
     # First, it fills the trainingset, than the validation and the testset
    # If the image exists in the dataset, we scale and resize it, then we add it to the set's inputs
    # Outputs are always in the Finding_Labels column
    if train_size > 0:
        if os.path.isfile(cfg.EXT_DATASET_PATH + "\\" + row['Image_Index']):
            img_scaled = readandscale(row)
            training_IN.append(img_scaled)
            training_OUT.append(row['Finding_Labels'])
            train_size -= 1
    elif valid_size > 0:
        if os.path.isfile(cfg.EXT_DATASET_PATH + "\\" + row['Image_Index']):
            img_scaled = readandscale(row)
            valid_IN.append(img_scaled)
            valid_OUT.append(row['Finding_Labels'])
            valid_size -= 1
    elif test_size > 0:
        if os.path.isfile(cfg.EXT_DATASET_PATH + "\\" + row['Image_Index']):
            img_scaled = readandscale(row)
            test_IN.append(img_scaled)
            test_OUT.append(row['Finding_Labels'])
            test_size -= 1
 #-------------------Model--------------------------
nb_classes = 15
# Random seed-ek beállítása
np.random.seed(1)
from tensorflow import set_random_seed
set_random_seed(2)
class TrainingHistory(Callback):
    # Tanulási folyamat elején létrehozunk egy-egy üres listát a kinyerni kívánt metrikák tárolása céljából.
    def on_train_begin(self, logs={}):
        # Hiba mértéke a tanító adatokon.
        self.losses = []
        # Hiba mértéke a validációs adatokon.
        self.valid_losses = []
        # A modell jóságát, pontosságát mérő mutatószám a tanító adatokon.
        self.accs = []
        # A modell jóságát, pontosságát mérő mutatószám a validációs adatokon.
        self.valid_accs = []
        # A tanítási fázisok sorszámozása.
        self.epoch = 0
     # Minden egyes tanítási fázis végén mentsük el, hogy hogyan teljesít aktuálisan a háló.
    def on_epoch_end(self, epoch, logs={}):
        self.losses.append(logs.get('loss'))
        self.valid_losses.append(logs.get('val_loss'))
        self.accs.append(logs.get('acc'))
        self.valid_accs.append(logs.get('val_acc'))
        self.epoch += 1
history = TrainingHistory()
model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(5,5), strides= 1, padding='same', activation='tanh',
                 input_shape=(256, 256,1)))
model.add(Conv2D(filters=64, kernel_size=(5,5), strides= 1, padding='same', activation='tanh'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Conv2D(filters=128, kernel_size=(5, 5), strides=1, padding='same', activation='tanh'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='tanh'))
model.add(Dense(nb_classes, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=0.001), metrics=['accuracy'])
 # Early stopping, amellyel figyeljük a validációs hibát (alap beállítás)
from keras.callbacks import EarlyStopping
patience=10
early_stopping=EarlyStopping(patience=patience, verbose=1)
 # Szintén a validációs hibát figyeljük, és elmentjük a legjobb modellt
from keras.callbacks import ModelCheckpoint
checkpointer=ModelCheckpoint(filepath='weights.hdf5', save_best_only=True, verbose=1)
 # a tanulálsi ráta (lr, learning rate) automatikus csökkentésére szolgáló függvény
from keras.callbacks import ReduceLROnPlateau
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=20, min_lr=10e-5)
model.fit(training_IN, training_OUT,
          # A tanulási folyamat során beállíthatjuk, hogy a tanító adatsorból hány elem kerüljön bele egy-egy
          # batch-be.
          batch_size=20,
          # Tanulási fázisok száma.
          nb_epoch=100,
          # Verbose paraméter a tanulás során közli azokat az információkat, amelyek mi szeretnénk kinyerni.
          # Értéke lehet 0, 1 és 2
          verbose=1,
          # A tanulással párhuzamosan a validáció is fut.
          validation_data=(valid_IN ,valid_OUT),
          # A korábbiakban már tárgyalt tanulást jellemző metrikákat a history nevű változóban szeretnénk tárolni.
          callbacks=[reduce_lr, checkpointer, early_stopping, history],
          # A bemenő adatokat keverje meg a program (alapbeállítás: True).
          shuffle= True)
plt.figure(figsize=(10, 5))
plt.title('Hiba mértéke')
plt.plot(np.arange(history.epoch), history.losses, color ='g', label='Hiba mértéke a tanító adatokon')
plt.plot(np.arange(history.epoch), history.valid_losses, color ='b', label='Hiba mértéke a validációs adatokon')
plt.legend(loc='upper right')
plt.xlabel('Tanulási iterációk száma')
plt.ylabel('y')
plt.grid(True)
plt.show()
 # a legjobb modell visszatöltése
from keras.models import load_model
model = load_model('weights.hdf5')
# teszt adatokkal prediktálás
preds=model.predict(test_IN)
# hiba számítása a teszt adatokon
from sklearn.metrics import mean_squared_error
test_mse = mean_squared_error(test_OUT,preds)
print("Test MSE: %f" % (test_mse))
model.summary()