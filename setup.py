import os
import sys
import numpy as np
import tensorflow as tf
import keras


if __name__ == "__main__":

    def is_google_colab():
        """Check if the environment is Google Colab."""
        try:
            import google.colab

            return True
        except ImportError:
            return False

    BASE_PATH = os.getcwd()

    if is_google_colab():
        from google.colab import drive

        BASE_PATH = os.path.join(BASE_PATH, "drive")
        drive.mount(BASE_PATH)

        DIR_PATH = os.path.join(BASE_PATH, "MyDrive", "da-svd")
        sys.path.append(DIR_PATH)
    else:
        DIR_PATH = os.path.join(BASE_PATH, "da-svd")
        sys.path.append(DIR_PATH)

    from util.constants import SEED, PATHS

    np.random.seed(SEED)
    tf.random.set_seed(SEED)
    keras.utils.set_random_seed(SEED)
