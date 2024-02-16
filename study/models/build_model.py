from ctypes import ArgumentError

from .models import (
    get_caltech101_model,
    get_catsdogs_model,
    get_fashion_model,
    get_mnist_model,
)

from util.trial_detail import TrialDetail


def build_model(trial_detail: TrialDetail):
    return _get_model(trial_detail)


def _get_model(trial_detail: TrialDetail):
    match trial_detail.dataset_name:
        case "mnist":
            return get_mnist_model(trial_detail)
        case "fashion_mnist":
            return get_fashion_model(trial_detail)
        case "cats_vs_dogs":
            return get_catsdogs_model(trial_detail)
        case "caltech101":
            return get_caltech101_model(trial_detail)
        case _:
            raise ArgumentError("Invalid Model Name")
