import numpy as np
import pandas as pd
import cv2

from tensorflow import keras
from tensorflow.keras.utils import Sequence, load_img, img_to_array

# https://mahmoudyusof.github.io/facial-keypoint-detection/data-generator/

class DataGenerator(Sequence):
    def __init__(self, csv_file, base_dir, output_size, shuffle=False, batch_size=10):
        """
        Initializes a data generator object
        :param csv_file: file in which image names and numeric labels are stored
        :param base_dir: the directory in which all images are stored
        :param output_size: image output size after preprocessing
        :param shuffle: shuffle the data after each epoch
        :param batch_size: The size of each batch returned by __getitem__
        """

        self.base_dir = base_dir
        self.output_size = output_size
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.on_epoch_end()

    def on_epoch_end(self):
        self.indices = np.arange(len(self.df))
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __len__(self):
        return int(len(self.df) / self.batch_size)

    def __getitem__(self, idx):
        pass