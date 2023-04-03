import numpy as np
import pandas as pd
import random
import os

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import Sequence, load_img, img_to_array
import sunpy.map

# https://mahmoudyusof.github.io/facial-keypoint-detection/data-generator/

class SunImgAEGenerator(tf.keras.utils.Sequence):
    def __init__(self, directory, batch_size, test_split=0.2, shuffle=True, noise_filter=False):
        
        # Get all the files in the directory
        self.directory = directory
        self.file_list = os.listdir(self.directory)
        self.file_list = list(map(lambda x: directory + x, self.file_list))

        # Remove noisy images if the flag is set 
        if noise_filter:
            noise_imgs = []
            with open("noisy_193A.csv", "r") as f:
                noise_imgs = f.readlines()
            
            noise_imgs = list(map(lambda x: x.strip(), noise_imgs))

            self.file_list = [f for f in self.file_list if f not in noise_imgs]

        # Shuffle data if the flag is set
        self.shuffle = shuffle
        if self.shuffle:
            random.shuffle(self.file_list)
        
        
        
        # Make train-test division
        self.train_list = self.file_list[:int(len(self.file_list)*(1-test_split))]
        self.test_list = self.file_list[int(len(self.file_list)*test_split):]

        self.batch_size = batch_size
        self.training = True
        self.take_all = False
    
    def sample(self, k):
        """
        Take a random sample from the dataset, primarily made for visualization and quick tests
        """

        idx = np.random.permutation(len(self.file_list))[:k]
        batch_maps = sunpy.map.Map([self.file_list[i] for i in idx])
        img_matrix = np.array(list(d.data for d in batch_maps))

        min_values = img_matrix.min(axis=2, keepdims=True).min(axis=1, keepdims=True)
        max_values = img_matrix.max(axis=2, keepdims=True).max(axis=1, keepdims=True)

        rg = max_values - min_values
        rg = np.fmax(rg, 1e-4*np.ones(rg.shape))

        print(rg.shape, min_values.shape, max_values.shape)
        img_matrix = (img_matrix-min_values) / rg

        return img_matrix

    def __len__(self):
        length = 0

        if self.take_all:
            length = len(self.file_list)
        elif self.training:
            length = len(self.train_list)
        else:
            length = len(self.test_list)
        
        return length//self.batch_size


    def __getitem__(self, idx):
        if self.take_all:
            batch_maps = sunpy.map.Map(self.file_list[idx * self.batch_size: (idx + 1) * self.batch_size])            
        elif self.training:
            batch_maps = sunpy.map.Map(self.train_list[idx * self.batch_size: (idx + 1) * self.batch_size])
        else:
            batch_maps = sunpy.map.Map(self.test_list[idx * self.batch_size: (idx + 1) * self.batch_size])
        img_matrix = np.array(list(d.data for d in batch_maps))

        min_values = img_matrix.min(axis=2, keepdims=True).min(axis=1, keepdims=True)
        max_values = img_matrix.max(axis=2, keepdims=True).max(axis=1, keepdims=True)

        rg = max_values - min_values
        rg = np.fmax(rg, 1e-4*np.ones(rg.shape))

        img_matrix = (img_matrix-min_values) / rg

        return img_matrix, img_matrix

    
    def on_epoch_end(self):
        if self.shuffle:
            random.shuffle(self.train_list)
            random.shuffle(self.test_list)

if __name__ == "__main__":
    datagen = SunImgAEGenerator("data/aia_193A/", 256, test_split=0.2, shuffle=True)

    print(datagen.__getitem__(80))
    print(len(datagen))