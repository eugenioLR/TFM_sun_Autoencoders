import numpy as np

import tensorflow as tf
from tensorflow import keras
from keras import layers


def gen_autoenc_model(latent_size, optim="adam", loss="mse"):
    input_img = keras.Input(shape=[256,256,1])

    x = input_img
    x = layers.Conv2D(8, 3, activation="relu", padding='same', strides=1)(x)
    x = layers.Conv2D(8, 3, activation="relu", padding='same', strides=1)(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(16, 3, activation="relu", padding='same', strides=1)(x)
    x = layers.Conv2D(16, 3, activation="relu", padding='same', strides=1)(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(32, 3, activation="relu", padding='same', strides=1)(x)
    x = layers.Conv2D(32, 3, activation="relu", padding='same', strides=1)(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Flatten()(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.1)(x)
    encoded = layers.Dense(latent_size, activation="sigmoid")(x)

    x = layers.Dense(256, activation="relu")(encoded)

    x = layers.Reshape([8, 8, 4])(x)
    x = layers.UpSampling2D()(x)

    x = layers.Conv2D(64, 3, activation="relu", padding='same', strides=1)(x)
    x = layers.Conv2DTranspose(64, 3, activation="relu", padding='same', strides=2)(x)

    x = layers.Conv2D(32, 3, activation="relu", padding='same', strides=1)(x)
    x = layers.Conv2DTranspose(32, 3, activation="relu", padding='same', strides=2)(x)

    x = layers.Conv2D(16, 3, activation="relu", padding='same', strides=1)(x)
    x = layers.Conv2DTranspose(16, 3, activation="relu", padding='same', strides=2)(x)

    x = layers.Conv2D(16, 3, activation="relu", padding='same', strides=1)(x)
    x = layers.Conv2DTranspose(1, 3, activation="relu", padding='same', strides=2)(x)

    decoded = x


    encoder = keras.Model(input_img, encoded)
    decoder = keras.Model(encoded, decoded)
    print(encoder.summary())
    decoder.summary()


    autoencoder = keras.Model(input_img, decoder(encoder(input_img)))
    autoencoder.compile(loss=loss, optimizer=optim, metrics=["mae"])
    autoencoder.summary()
    
    return autoencoder, encoder, decoder