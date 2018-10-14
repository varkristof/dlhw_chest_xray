import os
import pandas as pd
import cv2
from sklearn import preprocessing
import config as cfg


# Load the already processed csv file
entry = pd.read_csv("Data_Entry.csv")

# Calculate the set sizes
train_size = cfg.TRAIN_RATE * len(entry)
valid_size = cfg.VALID_RATE * len(entry)
test_size = cfg.TEST_RATE * len(entry)

# Inicialize the set inputs and outputs
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
# Rows int he csv file are already mixed
for index, row in entry.iterrows():

    # First, it fills the trainingset, than the validation and the testset
    # If the image exists in the dataset, we scales and resizes it, than we add it to the set's inputs
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

