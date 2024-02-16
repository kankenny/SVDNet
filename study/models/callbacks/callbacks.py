import os
import numpy as np
from textwrap import dedent
from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    # TensorBoard,
    EarlyStopping,
    LearningRateScheduler,
    Callback,
)

from util.constants import PATHS
from util.data_types.trial_detail import TrialDetail


def get_callbacks_list(trial_detail: TrialDetail, fold_num):
    dataset_name, augmentation_method, percentage_data, dist_rate = trial_detail

    checkpoint_cb = ModelCheckpoint(
        os.path.join(
            PATHS["BEST_MODELS"],
            f"{dataset_name}_{augmentation_method}_{percentage_data}_{dist_rate}.keras",
        ),
        save_best_only=True,
        monitor="val_loss",
    )
    # tensorboard_cb = TensorBoard(
    #     log_dir=_get_run_logdir(dataset_name, fold_num),
    #     profile_batch=(100, 200)
    # )
    earlystopping_cb = EarlyStopping(patience=20)
    lr_scheduler_cb = LearningRateScheduler(_scheduler)
    kfold_train_details_logger_cb = _KFoldTrainingDetailsLogger(trial_detail, fold_num)

    return [
        checkpoint_cb,
        # tensorboard_cb,
        earlystopping_cb,
        lr_scheduler_cb,
        kfold_train_details_logger_cb,
    ]


class _KFoldTrainingDetailsLogger(Callback):
    def __init__(self, trial_detail, fold_num):
        self.trial_detail = trial_detail
        self.fold_num = fold_num + 1

    def on_train_begin(self, logs=None):
        print(
            dedent(
                f"""
            \n\n{'*' * 80}\n\nSTART OF TRAINING - FOLD #{self.fold_num}:\n{self.trial_detail!r}\n\n{'*' * 80}\n\n"""
            )
        )

    def on_train_end(self, logs=None):
        print(
            dedent(
                f"""
            \n\n{'*' * 80}\n\nEND OF TRAINING - FOLD #{self.fold_num}:\n{self.trial_detail!r}\n\n{'*' * 80}\n\n"""
            )
        )


def _get_run_logdir(model_name, fold_num):
    return os.path.join(PATHS["TBOARD"], model_name, f"fold_{fold_num}")


def _scheduler(epoch, lr):
    if epoch < 10:
        return lr
    else:
        return lr * np.exp(-0.1)
