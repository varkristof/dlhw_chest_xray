import os
import io
import zipfile
from distutils.dir_util import copy_tree
import shutil
import config as cfg


# Extracts the given file recursively
def extract(filename):
    z = zipfile.ZipFile(filename)
    for f in z.namelist():
        # Get directory name from file
        dirname = cfg.DATASET_PATH + "\\" + os.path.splitext(f)[0]
        # Create new directory (temporarily)
        os.mkdir(dirname)
        # Read inner zip file into bytes buffer
        content = io.BytesIO(z.read(f))
        zip_file = zipfile.ZipFile(content)
        for i in zip_file.namelist():
            zip_file.extract(i, dirname)


# Merge the content of the extracted files into a common folder
# Deletes the separate folders
def merge(index):

    # Copy subdirectory
    fromdirectory = cfg.DATASET_PATH + "\images_" + index + "\images"
    todirectory = cfg.EXT_DATASET_PATH

    if os.path.exists(fromdirectory):
        copy_tree(fromdirectory, todirectory)
        shutil.rmtree(cfg.DATASET_PATH + "\images_" + index)


# Downloads the dataset from kaggle to DATASET_PATH via the kaggle api
# https://github.com/Kaggle/kaggle-api
download_command = "kaggle datasets download nih-chest-xrays/data -p" + cfg.DATASET_PATH

# Run the terminal command
os.system(download_command)

# Make the common folder (EXT_DATASET_PATH)
os.mkdir(cfg.EXT_DATASET_PATH)

extract(cfg.FILENAME)

# Possible folder indexes in the dataset
indexes = ["001", "002", "003", "004", "005", "006", "007", "008", "009", "010", "011", "012"]

# Move the content of every separate folder into the common one
for i in indexes:
    merge(i)

