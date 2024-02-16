import keras
from keras.layers import RandomZoom, RandomFlip, RandomRotation
from ctypes import ArgumentError

from compression_layer import Compression, RandomCompression


def get_augmentation_layer(augmentation_method, energy_factor, randomized=True):
    match augmentation_method:
        case "default":
            return _default_augmentations_layer()
        case "all":
            return _all_augmentations_layer(energy_factor, randomized)
        case "svd":
            return _compression_only_layer(energy_factor, randomized)
        case "none":
            return _no_augmentation_layer()
        case _:
            raise ArgumentError("Invalid augmentation method")


def _default_augmentations_layer():
    data_augmentation = keras.Sequential(
        [
            RandomFlip("horizontal"),
            RandomRotation(0.1),
            RandomZoom(0.1),
        ]
    )
    return data_augmentation


def _all_augmentations_layer(energy_factor, randomized):
    if randomized:
        compression = RandomCompression(
            energy_factor,
        )
    else:
        compression = Compression(
            energy_factor,
        )

    data_augmentation = keras.Sequential(
        [RandomFlip("horizontal"), RandomRotation(0.1), RandomZoom(0.1), compression]
    )
    return data_augmentation


def _compression_only_layer(energy_factor, randomized):
    if randomized:
        compression = RandomCompression(
            energy_factor,
        )
    else:
        compression = Compression(
            energy_factor,
        )
    return keras.Sequential([compression])


def _no_augmentation_layer():
    return keras.Sequential()
