import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

import numpy as np
import scipy
import scipy.stats
import pandas as pd
import astropy.units as u
import sunpy.map
import sunpy.visualization.colormaps as cm
import skimage
import glob
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
import random

import matplotlib.pyplot as plt
import seaborn as sns
import seaborn_image as isns
sns.set_style("darkgrid")

import tensorflow as tf
from tensorflow import keras
from keras import layers

import autoenc_model as aem
import data_generator as dg
import utils
import datetime

import json

batch_size = 48
gen_input = dg.MultiChannelAEGenerator("data/composite_data/", batch_size, test_split=0.2, shuffle=False, noise_filter=True)

latent_size = 768
optimizer = keras.optimizers.RMSprop(learning_rate=0.001)

loss_fn = "mse"

autoencoder, encoder, decoder = aem.gen_xception_autoenc_3c(latent_size, optim=optimizer, loss=loss_fn)


n_epochs = 200

history = autoencoder.fit(gen_input, epochs=n_epochs)


json_history_str = json.dumps(history.history)
with open("AE_xception_full.json", "w") as j:
    j.write(json_history_str)

autoencoder.save(f"autoencoder_{latent_size}_xception.h5")
encoder.save(f"encoder_{latent_size}_xception.h5")
decoder.save(f"decoder_{latent_size}_xception.h5")
