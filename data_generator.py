import numpy as np
import pandas as pd
# import cv2
import os

from tensorflow import keras
from tensorflow.keras.utils import Sequence, load_img, img_to_array

# https://mahmoudyusof.github.io/facial-keypoint-detection/data-generator/

class SunImgAEGenerator(tf.keras.utils.Sequence):

    def __init__(self, directory, batch_size):
        self.directory = directory
        self.file_list = os.listdir(self.directory)
        self.batch_size = batch_size

    def __len__(self):
        return len(self.list_files)

    def __getitem__(self, idx):
        batch_maps = sunpy.map.Map(self.file_list[idx * self.batch_size: (idx + 1) * self.batch_size])
        img_matrix = np.array(list(d.data for d in AIA193_2016))

        return np.array([
            resize(imread(file_name), (200, 200))
               for file_name in batch_x]), np.array(batch_y)