import numpy as np

import tensorflow as tf
from tensorflow import keras
from keras import layers

import utils
from utils import CylindricalPadding2D


def gen_autoenc_model_1c(latent_size, optim="adam", loss="mse", verbose=True):
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
    x = layers.Dropout(0.1)(x)
    encoded = layers.Dense(latent_size, activation="sigmoid")(x)

    x = layers.Dropout(0.1)(encoded)
    x = layers.Dense(1024, activation="relu")(x)
    x = layers.Reshape([8, 8, 16])(x)
    
    x = layers.Conv2D(64, 3, activation="relu", padding='same', strides=1)(x)
    x = layers.Conv2DTranspose(64, 3, activation="relu", padding='same', strides=2)(x)

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
    autoencoder = keras.Model(input_img, decoder(encoder(input_img)))
    autoencoder.compile(loss=loss, optimizer=optim, metrics=["mae"])

    if verbose:
        encoder.summary()
        decoder.summary()
        autoencoder.summary()
    
    return autoencoder, encoder, decoder

def gen_base_xception_autoenc_3c(latent_size, optim="adam", loss="mse", verbose=True):
    input_img = keras.Input(shape=[100,360,3])

    channel_axis = -1

    x = input_img
    
    x = layers.Conv2D(32, (3,3), strides=(2,2), use_bias=False)(x)
    x = layers.BatchNormalization(axis=channel_axis)(x)
    x = layers.Activation("relu")(x)

    x = layers.Conv2D(64, (3,3), use_bias=False)(x)
    x = layers.BatchNormalization(axis=channel_axis)(x)
    x = layers.Activation("relu")(x)


    residual = layers.Conv2D(128, (1,1), strides=(2,2), padding="same", use_bias=False)(x)
    residual = layers.BatchNormalization(axis=channel_axis)(residual)

    x = layers.SeparableConv2D(128, (3,3), padding="same", use_bias=False)(x)
    x = layers.BatchNormalization(axis=channel_axis)(x)

    x = layers.Activation("relu")(x)
    x = layers.SeparableConv2D(128, (3,3), padding="same", use_bias=False)(x)
    x = layers.BatchNormalization(axis=channel_axis)(x)

    x = layers.MaxPooling2D((3,3), strides=(2,2), padding="same")(x)

    x = layers.add([x, residual]) 


    residual = layers.Conv2D(256, (1,1), strides=(2,2), padding="same", use_bias=False)(x)
    residual = layers.BatchNormalization(axis=channel_axis)(residual)

    x = layers.SeparableConv2D(256, (3,3), padding="same", use_bias=False)(x)
    x = layers.BatchNormalization(axis=channel_axis)(x)

    x = layers.Activation("relu")(x)
    x = layers.SeparableConv2D(256, (3,3), padding="same", use_bias=False)(x)
    x = layers.BatchNormalization(axis=channel_axis)(x)

    x = layers.MaxPooling2D((3,3), strides=(2,2), padding="same")(x)

    x = layers.add([x, residual]) 


    residual = layers.Conv2D(728, (1,1), strides=(2,2), padding="same", use_bias=False)(x)
    residual = layers.BatchNormalization(axis=channel_axis)(residual)

    x = layers.SeparableConv2D(728, (3,3), padding="same", use_bias=False)(x)
    x = layers.BatchNormalization(axis=channel_axis)(x)

    x = layers.Activation("relu")(x)
    x = layers.SeparableConv2D(728, (3,3), padding="same", use_bias=False)(x)
    x = layers.BatchNormalization(axis=channel_axis)(x)

    x = layers.MaxPooling2D((3,3), strides=(2,2), padding="same")(x)

    x = layers.add([x, residual])

    
    for i in range(8):
        residual = x

        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(728, (3,3), padding="same", use_bias=False)(x)
        x = layers.BatchNormalization(axis=channel_axis)(x)

        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(728, (3,3), padding="same", use_bias=False)(x)
        x = layers.BatchNormalization(axis=channel_axis)(x)

        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(728, (3,3), padding="same", use_bias=False)(x)
        x = layers.BatchNormalization(axis=channel_axis)(x)

        x = layers.add([x, residual])
    

    residual = layers.Conv2D(728, (1,1), strides=(2,2), padding="same", use_bias=False)(x)
    residual = layers.BatchNormalization(axis=channel_axis)(residual)

    x = layers.Activation("relu")(x)
    x = layers.SeparableConv2D(728, (3,3), padding="same", use_bias=False)(x)
    x = layers.BatchNormalization(axis=channel_axis)(x)

    x = layers.Activation("relu")(x)
    x = layers.SeparableConv2D(728, (3,3), padding="same", use_bias=False)(x)
    x = layers.BatchNormalization(axis=channel_axis)(x)

    x = layers.MaxPooling2D((3,3), strides=(2,2), padding="same")(x)

    x = layers.add([x, residual])


    x = layers.SeparableConv2D(1536, (3,3), padding="same", use_bias=False)(x)
    x = layers.BatchNormalization(axis=channel_axis)(x)
    x = layers.Activation("relu")(x)

    x = layers.SeparableConv2D(2048, (3,3), padding="same", use_bias=False)(x)
    x = layers.BatchNormalization(axis=channel_axis)(x)
    x = layers.Activation("relu")(x)

    x = layers.GlobalAveragePooling2D()(x)
    
    encoded = layers.Dense(latent_size)(x)
    x = encoded

    x = layers.Dense(1024, activation="relu")(x)
    x = layers.Reshape([16, 16, 4])(x)
    x = layers.UpSampling2D()(x)

    x = layers.Conv2D(64, 3, activation="relu", padding='same', strides=1)(x)
    x = layers.Conv2DTranspose(64, 3, activation="relu", padding='same', strides=2)(x)

    x = layers.Conv2D(32, 3, activation="relu", padding='same', strides=1)(x)
    x = layers.Conv2DTranspose(32, 3, activation="relu", padding='same', strides=2)(x)

    x = layers.Conv2D(8, 3, activation="relu", padding='same', strides=1)(x)
    x = layers.Conv2DTranspose(3, 3, activation="relu", padding='same', strides=(1,2))(x)

    x = layers.Conv2D(8, 3, activation="relu", padding='same', strides=1)(x)
    x = layers.Conv2DTranspose(3, 3, activation="relu", padding='same', strides=(1,2))(x)

    x = layers.Cropping2D(((28,0),(152,0)))(x)
    x = layers.Conv2D(3, 3, activation="relu", padding='same', strides=1)(x)
    x = layers.Conv2D(3, 3, activation="relu", padding='same', strides=1)(x)

    decoded = x


    encoder = keras.Model(input_img, encoded)
    decoder = keras.Model(encoded, decoded)
    autoencoder = keras.Model(input_img, decoder(encoder(input_img)))
    autoencoder.compile(loss=loss, optimizer=optim, metrics=["mae"])

    if verbose:
        encoder.summary()
        decoder.summary()
        autoencoder.summary()
    
    return autoencoder, encoder, decoder

def gen_xception_autoenc_3c(latent_size, optim="adam", loss="mse", cylindical=True, verbose=True):
    

    input_img = keras.Input(shape=[100,360,3])

    channel_axis = -1

    x = input_img

    if cylindical:
        x = CylindricalPadding2D(5)(x)
    
    x = layers.Conv2D(32, (3,3), strides=(2,2), use_bias=False)(x)
    x = layers.BatchNormalization(axis=channel_axis)(x)
    x = layers.Activation("relu")(x)

    x = layers.Conv2D(64, (3,3), use_bias=False)(x)
    x = layers.BatchNormalization(axis=channel_axis)(x)
    x = layers.Activation("relu")(x)


    residual = layers.Conv2D(128, (1,1), strides=(2,2), padding="same", use_bias=False)(x)
    residual = layers.BatchNormalization(axis=channel_axis)(residual)

    x = layers.SeparableConv2D(128, (3,3), padding="same", use_bias=False)(x)
    x = layers.BatchNormalization(axis=channel_axis)(x)

    x = layers.Activation("relu")(x)
    x = layers.SeparableConv2D(128, (3,3), padding="same", use_bias=False)(x)
    x = layers.BatchNormalization(axis=channel_axis)(x)

    x = layers.MaxPooling2D((3,3), strides=(2,2), padding="same")(x)

    x = layers.add([x, residual]) 


    residual = layers.Conv2D(256, (1,1), strides=(2,2), padding="same", use_bias=False)(x)
    residual = layers.BatchNormalization(axis=channel_axis)(residual)

    x = layers.SeparableConv2D(256, (3,3), padding="same", use_bias=False)(x)
    x = layers.BatchNormalization(axis=channel_axis)(x)

    x = layers.Activation("relu")(x)
    x = layers.SeparableConv2D(256, (3,3), padding="same", use_bias=False)(x)
    x = layers.BatchNormalization(axis=channel_axis)(x)

    x = layers.MaxPooling2D((3,3), strides=(2,2), padding="same")(x)

    x = layers.add([x, residual]) 


    # comment After this
    residual = layers.Conv2D(728, (1,1), strides=(2,2), padding="same", use_bias=False)(x)
    residual = layers.BatchNormalization(axis=channel_axis)(residual)

    x = layers.SeparableConv2D(728, (3,3), padding="same", use_bias=False)(x)
    x = layers.BatchNormalization(axis=channel_axis)(x)

    x = layers.Activation("relu")(x)
    x = layers.SeparableConv2D(728, (3,3), padding="same", use_bias=False)(x)
    x = layers.BatchNormalization(axis=channel_axis)(x)

    x = layers.MaxPooling2D((3,3), strides=(2,2), padding="same")(x)

    x = layers.add([x, residual])

    
    for i in range(8):
        residual = x

        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(728, (3,3), padding="same", use_bias=False)(x)
        x = layers.BatchNormalization(axis=channel_axis)(x)

        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(728, (3,3), padding="same", use_bias=False)(x)
        x = layers.BatchNormalization(axis=channel_axis)(x)

        # Comment this
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(728, (3,3), padding="same", use_bias=False)(x)
        x = layers.BatchNormalization(axis=channel_axis)(x)

        x = layers.add([x, residual])
    

    residual = layers.Conv2D(256, (1,1), strides=(2,2), padding="same", use_bias=False)(x)
    residual = layers.BatchNormalization(axis=channel_axis)(residual)

    x = layers.Activation("relu")(x)
    x = layers.SeparableConv2D(256, (3,3), padding="same", use_bias=False)(x)
    x = layers.BatchNormalization(axis=channel_axis)(x)

    x = layers.Activation("relu")(x)
    x = layers.SeparableConv2D(256, (3,3), padding="same", use_bias=False)(x)
    x = layers.BatchNormalization(axis=channel_axis)(x)

    x = layers.MaxPooling2D((3,3), strides=(2,2), padding="same")(x)

    x = layers.add([x, residual])


    x = layers.SeparableConv2D(1536, (3,3), padding="same", use_bias=False)(x)
    x = layers.BatchNormalization(axis=channel_axis)(x)
    x = layers.Activation("relu")(x)

    x = layers.SeparableConv2D(2048, (3,3), padding="same", use_bias=False)(x)
    x = layers.BatchNormalization(axis=channel_axis)(x)
    x = layers.Activation("relu")(x)

    x = layers.GlobalAveragePooling2D()(x)
    
    encoded = layers.Dense(latent_size)(x)
    x = encoded

    x = layers.Dense(1024, activation="relu")(x)
    x = layers.Reshape([16, 16, 4])(x)
    x = layers.UpSampling2D()(x)

    x = layers.Conv2D(64, 3, activation="relu", padding='same', strides=1)(x)
    x = layers.Conv2DTranspose(64, 3, activation="relu", padding='same', strides=2)(x)

    x = layers.Conv2D(32, 3, activation="relu", padding='same', strides=1)(x)
    x = layers.Conv2DTranspose(32, 3, activation="relu", padding='same', strides=2)(x)

    x = layers.Conv2D(8, 3, activation="relu", padding='same', strides=1)(x)
    x = layers.Conv2DTranspose(3, 3, activation="relu", padding='same', strides=(1,2))(x)

    x = layers.Conv2D(8, 3, activation="relu", padding='same', strides=1)(x)
    x = layers.Conv2DTranspose(3, 3, activation="relu", padding='same', strides=(1,2))(x)

    x = layers.Cropping2D(((28,0),(152,0)))(x)

    decoded = x


    encoder = keras.Model(input_img, encoded)
    decoder = keras.Model(encoded, decoded)
    autoencoder = keras.Model(input_img, decoder(encoder(input_img)))
    autoencoder.compile(loss=loss, optimizer=optim, metrics=["mae"])

    if verbose:
        encoder.summary()
        decoder.summary()
        autoencoder.summary()
    
    return autoencoder, encoder, decoder


def test_cylinder():
    input_img = keras.Input(shape=[2,12,1])

    x = input_img
    x = CylindricalPadding2D(3)(x)

    model = keras.Model(input_img, x)

    test = np.array([range(12),range(12,12+12)]).reshape((1,2,12,1))
    result = model(test)[0,:,:,0]
    print(result)

def main():
    gen_xception_autoenc_3c(32)

if __name__ == "__main__":
    # test_cylinder()
    main()