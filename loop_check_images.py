import numpy as np
import scipy
import scipy.stats
import pandas as pd
import astropy.units as u
import sunpy.map
import sunpy.visualization.colormaps as cm
import cv2
from sklearn.cluster import KMeans

import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib as mpl
import seaborn as sns
import seaborn_image as isns
sns.set_style("darkgrid")
# isns.set_image(origin="upper")

import utils
import os
import glob
import random
import time
import aiofiles
import asyncio

import data_generator as dg

import getch

batch_size = 1
data_gen = dg.MultiChannelAEGenerator("data/composite_data/", batch_size, test_split=0.2, shuffle=False, noise_filter=True)
data_gen.take_all = True

noisy = []

# start = 3500
# start = [i.as_posix() for i in data_gen.file_list].index("data/composite_data/2019-02-15T02-30-14.npy")-1
start = [i.as_posix() for i in data_gen.file_list].index("data/composite_data/2015-05-14T01-31-14.npy")


#data/composite_data/2012-12-06T22-31-43.npy ????


for idx, data_point in enumerate(data_gen):
    if idx < start:
        continue 

    key = "-"
    
    isns.rgbplot(data_point[0][0], cmap=["sdoaia193", "sdoaia211", "seismic"], orientation="h", vmin=0, vmax=1)
    plt.show()    

    while key not in ["w", "e", "p", "b"]:
        print(f"{idx} Has noise? (w->No, e->Yes):")
        key = getch.getch()

    if key == 'e':
        print(f"Noise detected in file {data_gen.file_list[idx]}")
        noisy.append(data_gen.file_list[idx])
    elif key == 'w':
        print("No noise")
    elif key == 'p':
        print(f"Stopped. Last image: {data_gen.file_list[idx]}")
        break

with open("manual_noise.txt", "a") as f:
    [f.write("\n" + i.as_posix()) for i in noisy]

with open("aux.txt", "w") as f:
    f.write(str(idx))