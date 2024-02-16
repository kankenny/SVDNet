import keras
import tensorflow as tf
import tensorflow_datasets as tfds

from util.constants import HRSTC, DFLTS
from util.data_types.trial_detail import TrialDetail


IM_HEIGHT = 180
IM_WIDTH = 180
BUFFER_SZ = 1024


def load_dataset(trial_detail: TrialDetail):
    dataset_name, _, percentage_data, _ = trial_detail

    if dataset_name not in DFLTS["DATASETS"]:
        raise ValueError(f"Invalid Dataset Name: {dataset_name}")

    if dataset_name == "cats_vs_dogs":
        ds_train, ds_test = _awkward_dataset(percentage_data)
    else:
        ds_train, ds_test = tfds.load(
            dataset_name, split=["train", "test"], as_supervised=True
        )

        num_train_samples = int(len(ds_train) * percentage_data)

        ds_train = (
            ds_train.take(num_train_samples)
            .shuffle(BUFFER_SZ)
            .prefetch(tf.data.AUTOTUNE)
        )
        ds_test = ds_test.shuffle(BUFFER_SZ).prefetch(tf.data.AUTOTUNE)

    ds_train = ds_train.map(lambda image, label: (tf.cast(image, tf.float32), label))
    ds_test = ds_test.map(lambda image, label: (tf.cast(image, tf.float32), label))

    if dataset_name == "cats_vs_dogs" or dataset_name == "caltech101":
        ds_train = ds_train.map(
            lambda image, label: (_resize_image(image, IM_HEIGHT, IM_WIDTH), label)
        )
        ds_test = ds_test.map(
            lambda image, label: (_resize_image(image, IM_HEIGHT, IM_WIDTH), label)
        )

    return ds_train, ds_test


def split_dataset_kfold(ds_train, k):
    num_train_samples = len(ds_train)
    fold_size = num_train_samples // k

    fold_datasets = []
    for fold in range(k):
        start_index = fold * fold_size
        end_index = (fold + 1) * fold_size

        ds_val_fold = (
            ds_train.skip(start_index).take(fold_size).batch(HRSTC["BATCH_SZ"])
        )

        ds_train_fold_1 = ds_train.take(start_index).batch(HRSTC["BATCH_SZ"])
        ds_train_fold_2 = ds_train.skip(end_index).batch(HRSTC["BATCH_SZ"])
        ds_train_fold = ds_train_fold_1.concatenate(ds_train_fold_2)

        fold_datasets.append((ds_train_fold, ds_val_fold))

    return fold_datasets


def _awkward_dataset(percentage_data: float):
    """Cats-vs-Dogs tensorflow ds only has train split; boo!"""
    ds, ds_info = tfds.load(
        "cats_vs_dogs", split="train", with_info=True, as_supervised=True
    )

    num_train_samples = int(len(ds) * percentage_data)

    ds = ds.shuffle(BUFFER_SZ).prefetch(tf.data.AUTOTUNE)

    ds_train = ds.take(num_train_samples)
    ds_test = ds.skip(num_train_samples)

    return ds_train, ds_test


def _resize_image(img, height, width):
    resize_layer = keras.layers.Resizing(height, width)
    return resize_layer(img)
