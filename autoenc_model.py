import numpy as np

import tensorflow as tf
from tensorflow import keras
from keras import layers

import utils
from utils import CylindricalPadding2D

@keras.utils.register_keras_serializable()
class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        
        epsilon = tf.keras.backend.random_normal(shape=z_mean.shape)
        return z_mean + tf.exp(z_log_var) * epsilon


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

    z_mean = layers.Dense(latent_size, activation="sigmoid")(x)
    z_log_var = layers.Dense(latent_size, activation="sigmoid")(x)
    
    z = Sampling()([z_mean, z_log_var])


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

    encoder = keras.Model(input_img, z)
    decoder = keras.Model(encoded, decoded)
    autoencoder = keras.Model(input_img, decoder(encoder(input_img)))


    tf.keras.losses.KLDivergence()
    
    # Reconstruction loss compares inputs and outputs and tries to minimise the difference
    r_loss = original_dim * keras.losses.mse(visible, outpt)  # use MSE

    # KL divergence loss compares the encoded latent distribution Z with standard Normal distribution and penalizes if it's too different
    kl_loss =  -0.5 * K.sum(1 + z_log_sigma - K.square(z_mean) - K.exp(z_log_sigma), axis = 1)

    # The VAE loss is a combination of reconstruction loss and KL loss
    vae_loss = K.mean(r_loss + kl_loss)

    # Add loss to the model and compile it
    vae.add_loss(vae_loss)
    vae.compile(optimizer='adam')

    autoencoder.compile(loss=loss, optimizer=optim, metrics=["mae"])

    if verbose:
        encoder.summary()
        decoder.summary()
        autoencoder.summary()
    
    return autoencoder, encoder, decoder



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

def main():
    gen_xception_autoenc_3c(32)

if __name__ == "__main__":
    main()