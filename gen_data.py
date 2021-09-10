# -*- coding: utf-8 -*-

import cv2

IMG_SHAPE = (256, 256, 1)

# Filter images unfit for model training
# src: https://github.com/dandrino/terrain-erosion-3-ways/blob/58cdebf74bf9b0b08de61187fbd0ecebd4bb2386/generate_training_images.py#L16
def clean_sample(sample):
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

# Save the training images
def save_img(img, number):
    cv2.imwrite("data\\img\\" + str(number) + ".png", img)

# Process the images
def process_img(sample, number):
    # Check if it is a good sample, otherwise return and not save image
    sample = clean_sample(sample)
    if sample is None:
        return 0
    # Transform image through rotation and flipping to obtain and save 8 training images
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

# Counts the number of saved images
counter = 0
# Paths to unprocessed images
paths = []

# Iterate to build the paths of the training data
for i in range(1, 5001):
    path = str(i)
    while len(path) < 4:
        path = "0" + path
    path = "data\\" + path + "_h.png"
    paths.append(path)

# Iterates over the paths and loads images as grayscale
for path in paths:
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    # Cut out nine 256 by 256 subimages and processes them
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
