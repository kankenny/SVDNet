import os


def is_google_colab():
    """Check if the environment is Google Colab."""
    try:
        import google.colab

        return True
    except ImportError:
        return False


BASE_PATH = os.getcwd()

# Determine the base path and construct the dictionary accordingly
if is_google_colab():
    BASE_PATH = os.path.join(BASE_PATH, "drive", "MyDrive", "da-svd")
else:  # HPC or other environments
    BASE_PATH = os.path.join(BASE_PATH, "da-svd")
PATHS = {
    "BASE": BASE_PATH,
    "RESOURCES": os.path.join(BASE_PATH, "resources"),
    "PLOTS": os.path.join(BASE_PATH, "resources", "plots"),
    "TRIALS": os.path.join(BASE_PATH, "study", "trials"),
    "BEST_MODELS": os.path.join(BASE_PATH, "study", "best_state_models"),
    "TBOARD": os.path.join(BASE_PATH, "study", "trials", "tensorboard"),
}


HRSTC = {
    "NUM_FOLDS": 5,
    "MAX_ENERGY_FACTOR": 0.95,
    "MIN_ENERGY_FACTOR": 0.85,
    "SKIP_THRESHOLD": 0.1,
    "BATCH_SZ": 32,
    "MAX_EPS": 0.000001,
    "EPOCHS_MAP": {
        "mnist": 20,
        "fashion_mnist": 20,
        "cats_vs_dogs": 100,
        "caltech101": 200,
    },
}


DFLTS = {
    "DATASETS": ["mnist", "fashion_mnist", "cats_vs_dogs", "caltech101"],
    "DATA_PERCENTAGES": [1, 0.2],
    "AUGMENTATION_METHODS": ["default", "all", "svd", "none"],
    "ENERGY_FACTORS": [0.99, 0.95, 0.90],
}


SEED = 42
