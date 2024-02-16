import os
import sys
import pickle
from collections import namedtuple

from study.kfold_trial.kfold_trial import KFoldTrial

from util.constants import PATHS, DFLTS
from util.data_types.trial_detail import TrialDetail

TrialResult = namedtuple("TrialResult", "histories generalization_performance")


def do_specific_study(trial_detail: TrialDetail) -> None:
    trial = KFoldTrial(trial_detail)

    # Fit and predict are modularized and abstracted inside KFoldTrial
    histories = trial.fit()
    generalization_performance = trial.predict()

    trial_result = TrialResult(histories, generalization_performance)

    _save_trial_result(trial_result, trial_detail)


def experiment_1():
    """Determine if SVD image data augmentation provides any regularization"""
    aug_methods = [
        DFLTS["AUGMENTATION_METHODS"][2],  # SVD
        DFLTS["AUGMENTATION_METHODS"][3],
    ]  # None

    trial_details = [
        TrialDetail(dataset_name=ds, augmentation_method=aug, energy_factor=ef)
        for ds in DFLTS["DATASETS"]
        for aug in aug_methods
        for ef in DFLTS["ENERGY_FACTORS"]
    ]

    for trial_detail in trial_details:
        do_specific_study(trial_detail)


def experiment_2():
    """Determine if little data amplifies the SVD regularization effect (if any)"""
    aug_methods = [
        DFLTS["AUGMENTATION_METHODS"][2],  # SVD
        DFLTS["AUGMENTATION_METHODS"][3],
    ]  # None
    PARTIAL_TRAIN_DATA_PRC = DFLTS["DATA_PERCENTAGES"][1]  # 0.2

    trial_details = [
        TrialDetail(
            dataset_name=ds,
            augmentation_method=aug,
            percentage_data=PARTIAL_TRAIN_DATA_PRC,
            energy_factor=ef,
        )
        for ds in DFLTS["DATASETS"]
        for aug in aug_methods
        for ef in DFLTS["ENERGY_FACTORS"]
    ]

    for trial_detail in trial_details:
        do_specific_study(trial_detail)


def experiment_3():
    """
    default_augmentations = {random_zoom, random_rotate, random_horz_flip}
    Determine if SVD adds diversity along with the 'default' augmentations
    """
    aug_methods = [
        DFLTS["AUGMENTATION_METHODS"][0],  # DEFAULT
        DFLTS["AUGMENTATION_METHODS"][1],
    ]  # DEFAULT + SVD

    trial_details = [
        TrialDetail(dataset_name=ds, augmentation_method=aug, energy_factor=ef)
        for ds in DFLTS["DATASETS"]
        for aug in aug_methods
        for ef in DFLTS["ENERGY_FACTORS"]
    ]

    for trial_detail in trial_details:
        do_specific_study(trial_detail)


def _save_trial_result(trial_result, trial_detail):
    dataset_name, augmentation_method, percentage_data, energy_factor = trial_detail

    if augmentation_method not in {"all", "svd"}:
        energy_factor = 0

    output_path = os.path.join(PATHS["TRIALS"], dataset_name)
    file_path = f"{augmentation_method}_{percentage_data}_{energy_factor}.pkl"
    os.makedirs(output_path, exist_ok=True)

    with open(os.path.join(output_path, file_path), "wb") as f:
        pickle.dump(trial_result, f)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise ValueError("Expected one experiment integer argument: {1, 2, 3}")
        sys.exit(1)

    experiment = sys.argv[1]
    match experiment:
        case 1:
            experiment_1()
        case 2:
            experiment_2()
        case 3:
            experiment_3()
        case _:
            raise ValueError("Expected only one experiment integer argument: {1, 2, 3}")
            sys.exit(1)
