import numpy as np

import tensorflow as tf
from tensorflow import keras
from keras import layers
from tensorflow.keras import backend as K

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
    x = layers.Dense(latent_size, activation="sigmoid")(x)
    
    encoded = x

    x = layers.Dropout(0.1)(x)
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

def gen_VAE_model_1c(latent_size, optim="adam", loss="mse", VAE=False, verbose=True):
    input_img = keras.Input(shape=[256,256,1])

    # Define the encoder layers
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
    x = layers.Dense(latent_size, activation="sigmoid")(x)

    # Define the mean and variance layers
    z_mean = layers.Dense(latent_size)(x)
    z_log_var = layers.Dense(latent_size)(x)

    # Define the sampling function
    def sampling(args):
        z_mean, z_log_var = args
        epsilon = tf.keras.backend.random_normal(shape=(K.shape(z_mean)[0], latent_size), mean=0., stddev=1.0)
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

    # Define the sampling layer
    z = layers.Lambda(sampling)([z_mean, z_log_var])

    # Define the decoder layers
    decoder_input = layers.Input(shape=(latent_size,))

    x = decoder_input

    # x = layers.Dense(256*256, activation="relu")(x)
    # x = layers.Reshape([256, 256, 1])(x)

    x = layers.Dropout(0.1)(x)
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

    # Define the encoder and decoder models
    encoder = keras.Model(input_img, [z_mean, z_log_var, z], name='encoder')
    decoder = keras.Model(decoder_input, decoded, name='decoder')

    # Define the VAE model
    outputs = decoder(encoder(input_img)[2])
    vae = keras.Model(input_img, outputs, name='vae')

    # Define the VAE loss
    # reconstruction_loss = K.sum(K.mean(K.square(input_img - outputs), axis=((1,2))), axis=-1)
    reconstruction_loss = tf.reduce_mean(tf.reduce_sum(keras.losses.binary_crossentropy(input_img, outputs), axis=(1, 2)))

    kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
    kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=-1))

    vae_loss = reconstruction_loss + kl_loss
    vae.add_loss(vae_loss)

    # Compile the VAE model
    vae.compile(optimizer=optim)

    if verbose:
        encoder.summary()
        decoder.summary()
        vae.summary()
    
    return vae, encoder, decoder



def gen_xception_autoenc_polar_3c(latent_size, optim="adam", loss="mse", cylindical=True, verbose=True):
    input_img = keras.Input(shape=[100,360,3])

    channel_axis = -1

    x = input_img

    if cylindical:
        x = CylindricalPadding2D(3)(x)
    
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
    
    encoded = layers.Dense(latent_size, activation="tanh")(x)
    x = encoded

    x = layers.Dense(1024, activation="relu")(x)
    x = layers.Reshape([16, 16, 4])(x)

    x = layers.Conv2D(64, 3, activation="relu", padding='same', strides=1)(x)
    x = layers.Conv2DTranspose(64, 3, activation="relu", padding='same', strides=2)(x)

    x = layers.Conv2D(32, 3, activation="relu", padding='same', strides=1)(x)
    x = layers.Conv2DTranspose(32, 3, activation="relu", padding='same', strides=2)(x)

    x = layers.Conv2D(16, 3, activation="relu", padding='same', strides=1)(x)
    x = layers.Conv2DTranspose(16, 3, activation="relu", padding='same', strides=2)(x)

    x = layers.Conv2D(8, 3, activation="relu", padding='same', strides=1)(x)
    x = layers.Conv2DTranspose(8, 3, activation="relu", padding='same', strides=(1,2))(x)

    x = layers.Conv2D(3, 3, activation="relu", padding='same', strides=1)(x)
    x = layers.Conv2DTranspose(3, 3, activation="relu", padding='same', strides=(1,2))(x)

    x = layers.Conv2D(3, 3, activation="sigmoid", padding='same', strides=1)(x)

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

def gen_xception_autoenc_3c(latent_size, optim="adam", loss="mse", cylindical=True, verbose=True):
    input_img = keras.Input(shape=[204,204,3])

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
    
    encoded = layers.Dense(latent_size, activation="tanh")(x)
    x = encoded

    x = layers.Dense(1024, activation="relu")(x)
    x = layers.Reshape([8, 8, 16])(x)

    x = layers.Conv2D(64, 3, activation="relu", padding='same', strides=1)(x)
    x = layers.Conv2DTranspose(64, 3, activation="relu", padding='same', strides=2)(x)

    x = layers.Conv2D(32, 3, activation="relu", padding='same', strides=1)(x)
    x = layers.Conv2DTranspose(32, 3, activation="relu", padding='same', strides=2)(x)

    x = layers.Conv2D(16, 3, activation="relu", padding='same', strides=1)(x)
    x = layers.Conv2DTranspose(16, 3, activation="relu", padding='same', strides=2)(x)

    x = layers.Conv2D(8, 3, activation="relu", padding='same', strides=1)(x)
    x = layers.Conv2DTranspose(8, 3, activation="relu", padding='same', strides=2)(x)

    x = layers.Conv2D(3, 3, activation="relu", padding='same', strides=1)(x)
    x = layers.Conv2DTranspose(3, 3, activation="relu", padding='same', strides=2)(x)

    x = layers.Conv2D(3, 3, activation="tanh", padding='same', strides=1)(x)
    
    x = layers.Cropping2D(((26,26),(26,26)))(x)

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


def gen_xception_VAE_3c(latent_size, optim="adam", loss="mse", cylindical=True, verbose=True):

    input_img = keras.Input(shape=[204,204,3])

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
    
    encoded = layers.Dense(latent_size, activation="sigmoid")(x)

    # Define the mean and variance layers
    z_mean = layers.Dense(latent_size)(x)
    z_log_var = layers.Dense(latent_size)(x)

    # Define the sampling function
    def sampling(args):
        z_mean, z_log_var = args
        epsilon = tf.keras.backend.random_normal(shape=(K.shape(z_mean)[0], latent_size), mean=0., stddev=1.0)
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

    # Define the sampling layer
    z = layers.Lambda(sampling)([z_mean, z_log_var])

    # Define the decoder layers
    decoder_input = layers.Input(shape=(latent_size,))

    x = decoder_input

    x = layers.Dense(1024, activation="relu")(x)
    x = layers.Reshape([8, 8, 16])(x)

    x = layers.Conv2D(64, 3, activation="relu", padding='same', strides=1)(x)
    x = layers.Conv2DTranspose(64, 3, activation="relu", padding='same', strides=2)(x)

    x = layers.Conv2D(32, 3, activation="relu", padding='same', strides=1)(x)
    x = layers.Conv2DTranspose(32, 3, activation="relu", padding='same', strides=2)(x)

    x = layers.Conv2D(16, 3, activation="relu", padding='same', strides=1)(x)
    x = layers.Conv2DTranspose(16, 3, activation="relu", padding='same', strides=2)(x)

    x = layers.Conv2D(8, 3, activation="relu", padding='same', strides=1)(x)
    x = layers.Conv2DTranspose(8, 3, activation="relu", padding='same', strides=2)(x)

    x = layers.Conv2D(3, 3, activation="relu", padding='same', strides=1)(x)
    x = layers.Conv2DTranspose(3, 3, activation="relu", padding='same', strides=2)(x)

    x = layers.Conv2D(3, 3, activation="sigmoid", padding='same', strides=1)(x)
    
    x = layers.Cropping2D(((26,26),(26,26)))(x)

    decoded = x

    # Define the encoder and decoder models
    encoder = keras.Model(input_img, [z_mean, z_log_var, z], name='encoder')
    decoder = keras.Model(decoder_input, decoded, name='decoder')

    # Define the VAE model
    outputs = decoder(encoder(input_img)[2])
    vae = keras.Model(input_img, outputs, name='vae')

    # Define the VAE loss
    # reconstruction_loss = K.sum(K.mean(K.square(input_img - outputs), axis=((1,2))), axis=-1)
    reconstruction_loss = tf.reduce_mean(tf.reduce_sum(keras.losses.binary_crossentropy(input_img, outputs), axis=(1, 2)))

    kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
    kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=-1))

    vae_loss = reconstruction_loss + kl_loss
    vae.add_loss(vae_loss)

    # Compile the VAE model
    vae.compile(optimizer=optim)

    if verbose:
        encoder.summary()
        decoder.summary()
        vae.summary()
    
    return vae, encoder, decoder


def main():
    gen_xception_autoenc_3c(32)

if __name__ == "__main__":
    main()