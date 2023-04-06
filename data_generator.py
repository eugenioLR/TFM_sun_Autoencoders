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
    
    @staticmethod
    def normalize(data_matrix):
        data_matrix = np.clip(data_matrix, 0, 5000)

        min_values = data_matrix.min(axis=2, keepdims=True).min(axis=1, keepdims=True)
        max_values = data_matrix.max(axis=2, keepdims=True).max(axis=1, keepdims=True)

        rg = max_values - min_values
        rg = np.fmax(rg, 1e-4)

        data_matrix = (data_matrix-min_values) / rg

        return data_matrix

    
    def sample(self, k):
        """
        Take a random sample from the dataset, primarily made for visualization and quick tests
        """

        idx = np.random.permutation(len(self.file_list))[:k]
        batch_maps = sunpy.map.Map([self.file_list[i] for i in idx])
        img_matrix = np.array(list(d.data for d in batch_maps))

        img_matrix = self.normalize(img_matrix)

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

        img_matrix = self.normalize(img_matrix)

        return img_matrix, img_matrix
    
    def on_epoch_end(self):
        if self.shuffle:
            random.shuffle(self.train_list)
            random.shuffle(self.test_list)


class SunImgAE3CGenerator(SunImgAEGenerator):
    @staticmethod
    def normalize(data_matrix):
        data_matrix[:,:,:,0] = np.clip(data_matrix[:,:,:,0], 0, 5000)
        data_matrix[:,:,:,1] = np.clip(data_matrix[:,:,:,1], 0, 3000)
        # data_matrix[:,:,:,2] = np.clip(data_matrix[:,:,:,2], -400, 400)

        min_values = np.nanmin(np.nanmin(data_matrix, axis=2, keepdims=True), axis=1, keepdims=True)
        max_values = np.nanmax(np.nanmax(data_matrix, axis=2, keepdims=True), axis=1, keepdims=True)

        rg = max_values - min_values
        rg = np.fmax(rg, 1e-4)

        data_matrix = (data_matrix-min_values) / rg

        # Normalize HMI magnetogram data differently
        data_matrix[:,:,:,2] = np.clip(data_matrix[:,:,:,2], -400, 400)
        hmi_max_values = np.nanmax(np.nanmax(np.abs(data_matrix[:,:,:,2]), axis=2, keepdims=True), axis=1, keepdims=True)
        data_matrix[:,:,:,2] = ((data_matrix[:,:,:,2] / hmi_max_values) + 1)/2

        data_matrix[np.isnan(data_matrix)] = 0

        return data_matrix    

    def sample(self, k):
        """
        Take a random sample from the dataset, primarily made for visualization and quick tests
        """

        idx = np.random.permutation(len(self.file_list))[:k]
        batch_maps = sunpy.map.Map([self.file_list[i] for i in idx])
        img_matrix = np.array(list(d.data for d in batch_maps))

        img_matrix = self._normalize(img_matrix)

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

        img_matrix = self.normalize(img_matrix)

        return img_matrix, img_matrix

    
    def on_epoch_end(self):
        if self.shuffle:
            random.shuffle(self.train_list)
            random.shuffle(self.test_list)

if __name__ == "__main__":
    datagen = SunImgAEGenerator("data/aia_193A/", 256, test_split=0.2, shuffle=True)

    print(datagen.__getitem__(80))
    print(len(datagen))