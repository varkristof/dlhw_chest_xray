import os
import pandas as pd
import numpy as np
import cv2


def create_directory(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)


#DATASET_PATH = "E:\chest_dataset\images_001\images\\"
DATASET_PATH = ""
IMG_SIZE = 258
TRAIN_RATE = 0.5
VALID_RATE = 0.2
TEST_RATE = 0.3
entry = pd.read_csv("Data_Entry.csv")

train_size = TRAIN_RATE * len(entry)
valid_size = VALID_RATE * len(entry)
test_size = TEST_RATE * len(entry)

train_IN = []
valid_IN = []
test_IN = []

train_OUT = []
valid_OUT = []
test_OUT = []

patternDel = "|"
for index, row in entry.iterrows():
    if patternDel not in row['Finding_Labels'] and train_size > 0:
        if os.path.isfile(DATASET_PATH + row['Image_Index']):
            img_src = cv2.imread(DATASET_PATH + row['Image_Index'], cv2.IMREAD_GRAYSCALE)
            train_IN.append(cv2.resize(img_src, (IMG_SIZE, IMG_SIZE)))
            train_OUT.append(row['Finding_Labels'])
            train_size -= 1

    elif valid_size > 0:
        if os.path.isfile(DATASET_PATH + row['Image_Index']):
            img_src = cv2.imread(DATASET_PATH + row['Image_Index'], cv2.IMREAD_GRAYSCALE)
            valid_IN.append(cv2.resize(img_src, (IMG_SIZE, IMG_SIZE)))
            valid_OUT.append(row['Finding_Labels'])
            valid_size -= 1

    elif test_size > 0:
        if os.path.isfile(DATASET_PATH + row['Image_Index']):
            img_src = cv2.imread(DATASET_PATH + row['Image_Index'], cv2.IMREAD_GRAYSCALE)
            test_IN.append(cv2.resize(img_src, (IMG_SIZE, IMG_SIZE)))
            test_OUT.append(row['Finding_Labels'])
            test_size -= 1
