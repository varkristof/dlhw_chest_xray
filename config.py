
#The downloader script going to download the (compressed) dataset into this folder
DATASET_PATH = "res"

#The unzipped dataset is in this folder
EXT_DATASET_PATH = DATASET_PATH + "/images"

#Path and name of the compressed dataset
FILENAME = DATASET_PATH + "/data.zip"

#Size of the images in px
#IMG_SIZE x IMG_SIZE
IMG_SIZE = 256

#Rate of the train-, validation- and testsets
#The sum of them have to be one
TRAIN_RATE = 0.5
VALID_RATE = 0.2
TEST_RATE = 0.3
