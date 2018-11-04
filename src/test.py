import os

from keras.models import load_model
import numpy as np
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error
from sklearn.metrics import confusion_matrix
import seaborn as sns

import cv2
import config as cfg
import pandas as pd

# Load the already processed csv file
entry = pd.read_csv("data_first_folder.csv")

# Calculate the set sizes
train_size = cfg.TRAIN_RATE * len(entry)
valid_size = cfg.VALID_RATE * len(entry)
test_size = cfg.TEST_RATE * len(entry)

# Initialize the set inputs and outputs
samples=np.zeros(len(entry), dtype=[('input', float, (256,256)), ('output', float, 15)])


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
    if os.path.isfile(cfg.EXT_DATASET_PATH + "/" + row['Image_Index']):
        img_scaled = readandscale(row)
        samples[index] = img_scaled, row[7:]

train = samples[0:int(len(entry)*(1-cfg.VALID_RATE-cfg.TEST_RATE))]
valid = samples[int(len(entry)*(1-cfg.VALID_RATE-cfg.TEST_RATE)):int(len(entry)*(1-cfg.TEST_RATE))]
test = samples[int(len(entry)*(1-cfg.TEST_RATE)):]

train_x = np.reshape(train['input'], (len(train), 256, 256, 1))
valid_x = np.reshape(valid['input'], (len(valid), 256, 256, 1))
test_x = np.reshape(test['input'], (len(test), 256, 256, 1))




# a legjobb modell visszatöltése
model = load_model('weights.hdf5')

# teszt adatokkal prediktálás
preds=model.predict(test_x)

# hiba számítása a teszt adatokon
test_mse = mean_squared_error(test['output'], preds)
print("Test MSE: %f" % (test_mse))
model.summary()

conf = confusion_matrix(test['output'], np.argmax(preds, axis=1))
sns.set()
sns.heatmap(conf)