# -*- coding: utf-8 -*-

import tensorflow as tf
import cv2
import numpy as np

IMG_SHAPE = (256, 256, 1)


def clean_sample(sample):
    # Get rid of "out-of-bounds" magic values.
    sample[sample == np.finfo('float32').min] = 0.0

    # Ignore any samples with NaNs, for one reason or another.
    if np.isnan(sample).any():
        return None

    # Only accept values that span a given range. This is to capture more
    # mountainous samples.
    if (sample.max() - sample.min()) < 40:
        return None

    # Filter out samples for which a significant portion is within a small
    # threshold from the minimum value. This helps filter out samples that
    # contain a lot of water.
    near_min_fraction = (sample < (sample.min() + 8)).sum() / sample.size
    if near_min_fraction > 0.2:
        return None

    return sample


def save_img(img, number):
    cv2.imwrite("data\\img\\" + str(number) + ".png", img)


def process_img(sample, number):
    sample = clean_sample(sample)
    if sample is None:
        return 0
    save_img(sample, number)
    save_img(cv2.rotate(sample, cv2.ROTATE_90_CLOCKWISE), number + 1)
    save_img(cv2.rotate(sample, cv2.ROTATE_90_COUNTERCLOCKWISE), number + 2)
    save_img(cv2.rotate(sample, cv2.ROTATE_180), number + 3)
    sample = cv2.flip(sample, 1)
    save_img(sample, number + 4)
    save_img(cv2.rotate(sample, cv2.ROTATE_90_CLOCKWISE), number + 5)
    save_img(cv2.rotate(sample, cv2.ROTATE_90_COUNTERCLOCKWISE), number + 6)
    save_img(cv2.rotate(sample, cv2.ROTATE_180), number + 7)
    return 8


counter = 0
paths = []

for i in range(1, 5001):
    path = str(i)
    while len(path) < 4:
        path = "0" + path
    path = "data\\" + path + "_h.png"
    paths.append(path)

for path in paths:
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    image_train = image[:256, :256]
    counter += process_img(image_train, counter)

    image_train = image[256:, :256]
    counter += process_img(image_train, counter)

    image_train = image[:256, 256:]
    counter += process_img(image_train, counter)

    image_train = image[256:, 256:]
    counter += process_img(image_train, counter)

    image_train = image[128:384, :256]
    counter += process_img(image_train, counter)

    image_train = image[128:384:, 256:]
    counter += process_img(image_train, counter)

    image_train = image[:256, 128:384]
    counter += process_img(image_train, counter)

    image_train = image[256:, 128:384]
    counter += process_img(image_train, counter)

    image_train = image[128:384, 128:384]
    counter += process_img(image_train, counter)

print("Images added: " + str(counter))
