
import os

import cv2
import pandas as pd
from sklearn import preprocessing

IMG_SIZE = 256

# Loads the already processed csv file
entry = pd.read_csv("entry_data_edited.csv")

TRAIN_RATE = 0.5
VALID_RATE = 0.2
TEST_RATE = 0.3

# Calculates the number of training, validation, testing images
train_size = TRAIN_RATE * len(entry)
valid_size = VALID_RATE * len(entry)
test_size = TEST_RATE * len(entry)

# Reads an image from the dataset, converts it to grayscale, resizes it and saves it its folder
def readandscale(im_row, num):
    # Reads out the name of the picture (Image_Index) and reads the image in grayscale
    src = cv2.imread("images/" + im_row['Image_Index'], cv2.IMREAD_GRAYSCALE)
    # Resizes and saves the image
    resized = cv2.resize(src, (IMG_SIZE, IMG_SIZE))
    if num <= train_size:
        cv2.imwrite("train/" + im_row['Image_Index'], resized)
    if (num > train_size) and (num <= train_size+valid_size):
        cv2.imwrite("valid/" + im_row['Image_Index'], resized)
    if (num > train_size+valid_size) and (num <= len(entry)):
        cv2.imwrite("test/" + im_row['Image_Index'], resized)


# Iterates through the rows of the dataframe
# Rows in the csv file are already mixed
for index, row in entry.iterrows():
    # If the image exists in the dataset, we scale and resize it

    if os.path.isfile("images/" + row['Image_Index']):
        readandscale(row, index)
