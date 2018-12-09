# dlhw_chest_xray

### Software environment
We used Python 3.6 , Keras 2.2.4 and TensorFlow 1.11.0 as backend.

### Reproduce the process
First, at config.py you need to modify the DATASET_PATH variable regarding to your preferencies. The other files will use this path. You find also these parameters in this file:
  - IMG_SIZE - Size of the images after the resizing
  - TRAIN_RATE, VALID_RATE, TEST_RATE - The rate of the training, validation and testing dataset regarding to the whole dataset
  
  After the you need to run the Python files in this order:
  1. download_decompress.py - It will download the [NIH Chest X-ray Dataset](https://www.kaggle.com/nih-chest-xrays/data/home), and decompress it.
  2. preproCsv.py - It will create the CSV files for the the image preprocessing, training and testing.
  3. prepro.py - It will resize, and separate in different folders the images.
  4. train.py - You find here the model, and the code of the training. 
  5. test.py - You find here the testing part.
