import numpy as np
import pandas as pd
import random
import os

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import Sequence, load_img, img_to_array
import sunpy.map
import pathlib
import utils

from skimage.filters import gaussian


# https://mahmoudyusof.github.io/facial-keypoint-detection/data-generator/

class SunImgAEGenerator(tf.keras.utils.Sequence):
    def __init__(self, directory, batch_size, test_split=0.2, shuffle=True, noise_filter=False):
        self.file_list = list(pathlib.Path(directory).iterdir())

        # Remove noisy images if the flag is set 
        if noise_filter:
            noise_imgs = []
            with open("noisy_193A.csv", "r") as f:
                noise_imgs = f.readlines()
            
            noise_imgs = [pathlib.Path(i.strip()).stem for i in noise_imgs]

            self.file_list = [f for f in self.file_list if f.stem not in noise_imgs]

        # Shuffle data if the flag is set
        self.shuffle = shuffle
        if self.shuffle:
            random.shuffle(self.file_list)        
        
        # Make train-test division
        split_point = int(len(self.file_list)*test_split)
        self.train_list = self.file_list[:len(self.file_list)-split_point]
        self.test_list = self.file_list[split_point:]

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
        img_matrix = np.array([d.data for d in batch_maps])
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
        img_matrix = np.array([d.data for d in batch_maps])

        img_matrix = self.normalize(img_matrix)

        return img_matrix, img_matrix
    
    def on_epoch_end(self):
        if self.shuffle:
            random.shuffle(self.train_list)
            random.shuffle(self.test_list)
    
class AIA193Generator(SunImgAEGenerator):
    @staticmethod
    def normalize(data_matrix):
        data_matrix = np.clip(data_matrix, 0, 5000)

        min_values = np.nanmin(np.nanmin(data_matrix, axis=2, keepdims=True), axis=1, keepdims=True)
        max_values = np.nanmax(np.nanmax(data_matrix, axis=2, keepdims=True), axis=1, keepdims=True)

        rg = max_values - min_values
        rg = np.fmax(rg, 1e-4)

        data_matrix = (data_matrix-min_values) / rg

        return data_matrix

class AIA211Generator(SunImgAEGenerator):
    @staticmethod
    def normalize(data_matrix):
        data_matrix = np.clip(data_matrix, 0, 3000)

        min_values = np.nanmin(np.nanmin(data_matrix, axis=2, keepdims=True), axis=1, keepdims=True)
        max_values = np.nanmax(np.nanmax(data_matrix, axis=2, keepdims=True), axis=1, keepdims=True)

        rg = max_values - min_values
        rg = np.fmax(rg, 1e-4)

        data_matrix = (data_matrix-min_values) / rg

        return data_matrix


class HMImGenerator(SunImgAEGenerator):
    @staticmethod
    def normalize(data_matrix, nan_fill=0):
        data_matrix = np.clip(data_matrix, -100, 100)

        max_values = np.nanmax(np.nanmax(np.abs(data_matrix), axis=2, keepdims=True), axis=1, keepdims=True)

        data_matrix = (data_matrix / max_values + 1)/2

        data_matrix[np.isnan(data_matrix)] = nan_fill

        return data_matrix
    
    @staticmethod
    def apply_filter(data_matrix, kernel_size=3, sigma=2, cval=0):
        # gaussian(utils.abs_max_filter(2*x - 1, absmax_size)/2 + 0.5, gauss_sigma)

        data_matrix_norm = 2*data_matrix - 1

        data_matrix_norm = utils.abs_max_filter_par(data_matrix_norm, kernel_size, cval=cval)
        data_matrix_norm = gaussian(data_matrix_norm, sigma=sigma, channel_axis=0, cval=cval)

        data_matrix = (data_matrix_norm + 1)/2

        return data_matrix

class PolarHMImGenerator(HMImGenerator):    
    @staticmethod
    def apply_filter(data_matrix, kernel_size=3, sigma=2, cval=0):
        data_matrix_c = np.asarray([utils.polar_to_cartesian(img, cval=0.5) for img in data_matrix])
        data_matrix_c = HMImGenerator.apply_filter(data_matrix_c, kernel_size, sigma, cval)
        data_matrix = np.asarray([utils.cartesian_to_polar(img) for img in data_matrix_c])

        return data_matrix



class MultiChannelAEGenerator(SunImgAEGenerator):
    def __init__(self, directory, batch_size, test_split=0.2, shuffle=True, noise_filter=False, filter_hmi=True, filter_sigma=2.5):
        super().__init__(directory, batch_size, test_split, shuffle, noise_filter)
        self.filter_hmi = filter_hmi
        self.filter_sigma = filter_sigma


    @staticmethod
    def normalize(data_matrix):
        data_matrix[:,:,:,0] = np.clip(data_matrix[:,:,:,0], 0, 5000)
        data_matrix[:,:,:,1] = np.clip(data_matrix[:,:,:,1], 0, 3000)
        data_matrix[:,:,:,2] = np.clip(data_matrix[:,:,:,2], -100, 100)

        min_values = np.nanmin(np.nanmin(data_matrix, axis=2, keepdims=True), axis=1, keepdims=True)
        max_values = np.nanmax(np.nanmax(data_matrix, axis=2, keepdims=True), axis=1, keepdims=True)

        rg = max_values - min_values
        rg = np.fmax(rg, 1e-4)

        data_matrix_norm = (data_matrix-min_values) / rg

        # Normalize HMI magnetogram data differently
        hmi_max_values = np.nanmax(np.nanmax(np.abs(data_matrix[:,:,:,2]), axis=2, keepdims=True), axis=1, keepdims=True)
        data_matrix_norm[:,:,:,2] = ((data_matrix[:,:,:,2] / hmi_max_values) + 1)/2
        
        hmi_nans = np.isnan(data_matrix_norm[:,:,:,2])

        data_matrix_norm[np.isnan(data_matrix_norm)] = 0
        data_matrix_norm[hmi_nans, 2] = 0.5

        return data_matrix_norm   

    def sample(self, k):
        """
        Take a random sample from the dataset, primarily made for visualization and quick tests
        """

        idx = np.random.permutation(len(self.file_list))[:k]

        batch_files = [self.file_list[i] for i in idx]
        img_matrix = np.empty([len(batch_files), 256, 256, 3])
        for idx, data_file in enumerate(batch_files):
            data_point = np.load(data_file)
            img_matrix[idx] = data_point

        img_matrix = self.normalize(img_matrix)

        if self.filter_hmi:
            img_matrix[:,:,:,2] = HMImGenerator.apply_filter(img_matrix[:,:,:,2], 4, self.filter_sigma)

        return img_matrix

    def __getitem__(self, idx):
        if self.take_all:
            batch_files = self.file_list[idx * self.batch_size: (idx + 1) * self.batch_size]
        elif self.training:
            batch_files = self.train_list[idx * self.batch_size: (idx + 1) * self.batch_size]
        else:
            batch_files = self.test_list[idx * self.batch_size: (idx + 1) * self.batch_size]

        img_matrix = np.empty([len(batch_files), 256, 256, 3])
        for idx, data_file in enumerate(batch_files):
            data_point = np.load(data_file)
            img_matrix[idx] = data_point

        img_matrix = self.normalize(img_matrix)

        if self.filter_hmi:
            img_matrix[:,:,:,2] = HMImGenerator.apply_filter(img_matrix[:,:,:,2], 4, self.filter_sigma)

        return img_matrix, img_matrix

class PolarMultiChannelAEGenerator(MultiChannelAEGenerator):
    def sample(self, k):
        """
        Take a random sample from the dataset, primarily made for visualization and quick tests
        """

        idx = np.random.permutation(len(self.file_list))[:k]

        batch_files = [self.file_list[i] for i in idx]
        img_matrix = np.empty([len(batch_files), 100, 360, 3])
        for idx, data_file in enumerate(batch_files):
            data_point = np.load(data_file)
            img_matrix[idx] = data_point

        img_matrix = self.normalize(img_matrix)

        return img_matrix

    def __getitem__(self, idx):
        if self.take_all:
            batch_files = self.file_list[idx * self.batch_size: (idx + 1) * self.batch_size]
        elif self.training:
            batch_files = self.train_list[idx * self.batch_size: (idx + 1) * self.batch_size]
        else:
            batch_files = self.test_list[idx * self.batch_size: (idx + 1) * self.batch_size]

        img_matrix = np.empty([len(batch_files), 100, 360, 3])
        for idx, data_file in enumerate(batch_files):
            data_point = np.load(data_file)
            img_matrix[idx] = data_point

        img_matrix = self.normalize(img_matrix)

        return img_matrix, img_matrix

def test_1c_gen():
    datagen = SunImgAEGenerator("data/aia_193A/", 256, test_split=0.2, shuffle=True, noise_filter=True)
    datagen.__getitem__(8)
    datagen.sample(2)

def test_3c_gen():
    datagen = SunImgAE3CGenerator("data/composite_data/", 256, test_split=0.2, shuffle=True, noise_filter=True)
    datagen.__getitem__(8)
    datagen.sample(2)

if __name__ == "__main__":
    test_1c_gen()
    test_3c_gen()