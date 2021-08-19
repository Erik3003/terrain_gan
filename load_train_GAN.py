# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow import keras
from wgan_gp import WGAN, generator_loss, discriminator_loss, train_images, BATCH_SIZE, generator_optimizer, \
    discriminator_optimizer, noise_dim

# number of epochs you want to train
epochs = 6

# Epoch of the Generator version - change this to start for training a specific model in models folder
startEpoch = 1

g_model = keras.models.load_model('./models/Generator_{epoch}.h5'.format(epoch=startEpoch))
d_model = keras.models.load_model('./models/Discriminator.h5')


class GANMonitorNEW(keras.callbacks.Callback):
    def __init__(self, num_img=5, latent_dim=128):
        self.num_img = num_img
        self.latent_dim = latent_dim
        self.random_latent_vectors = tf.random.normal(shape=(self.num_img, self.latent_dim), seed=3342934)

    def on_epoch_end(self, epoch, logs=None):
        generated_images = self.model.generator(self.random_latent_vectors)
        generated_images = (generated_images * 127.5) + 127.5

        for i in range(self.num_img):
            img = generated_images[i].numpy()
            img = keras.preprocessing.image.array_to_img(img)
            img.save("output\\{epoch}_heightmap_{i}.png".format(i=i, epoch=epoch))

        self.model.generator.save("retrained_models\\Generator_{epoch}.h5".format(epoch=epoch + startEpoch + 1))
        self.model.discriminator.save("retrained_models\\Discriminator.h5")


newGanModel = WGAN(
    discriminator=d_model,
    generator=g_model,
    latent_dim=noise_dim,
    discriminator_extra_steps=5,
)

# Compile the WGAN model.
newGanModel.compile(
    d_optimizer=discriminator_optimizer,
    g_optimizer=generator_optimizer,
    g_loss_fn=generator_loss,
    d_loss_fn=discriminator_loss,
)

# Instantiate the customer `GANMonitor` Keras callback.
cbk = GANMonitorNEW(num_img=6, latent_dim=noise_dim)

# Start training the model.
newGanModel.fit(train_images, batch_size=BATCH_SIZE, epochs=epochs, callbacks=[cbk])
newGanModel.fit(train_images, batch_size=int(BATCH_SIZE / 2), epochs=epochs, callbacks=[cbk])
newGanModel.fit(train_images, batch_size=int(BATCH_SIZE / 4), epochs=epochs, callbacks=[cbk])
newGanModel.fit(train_images, batch_size=int(BATCH_SIZE / 8), epochs=epochs, callbacks=[cbk])
newGanModel.fit(train_images, batch_size=int(BATCH_SIZE / 16), epochs=epochs, callbacks=[cbk])