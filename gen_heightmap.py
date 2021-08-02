# -*- coding: utf-8 -*-
import random

import numpy as np
import cv2
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

num_img = 5
noise_dim = 128

generator = keras.models.load_model("Generator.h5")
random_latent_vectors = tf.random.normal(shape=(num_img, noise_dim))

generated_images = generator.predict(random_latent_vectors)
generated_images = (generated_images * 127.5) + 127.5

for i in range(num_img):
    img = generated_images[i]
    img = keras.preprocessing.image.array_to_img(img)
    img.save("output\\generated_img_{i}.png".format(i=i))
