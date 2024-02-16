import numpy as np
import tensorflow as tf


def normalize_img(img):
    min_vals = np.min(img, axis=(0, 1), keepdims=True)
    max_vals = np.max(img, axis=(0, 1), keepdims=True)

    normalized_array = (img - min_vals) / (max_vals - min_vals + 1e-8)

    return normalized_array


def absolute_err(mat1, mat2):
    return np.absolute((mat1 - mat2))


def mean_absolute_err(mat1, mat2):
    absolute_errors = absolute_err(mat1, mat2)
    return np.mean(absolute_errors)


def total_absolute_err(mat1, mat2):
    absolute_errors = absolute_err(mat1, mat2)
    return np.sum(absolute_errors)


def load_image(file_path, channels=3):
    image = tf.io.read_file(file_path)
    image = tf.image.decode_image(image, channels=3) / 255
    return image


def elementwise_err(mat1, mat2):
    return mat1 - mat2
