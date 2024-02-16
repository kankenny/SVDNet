from study.models import build_model
from study.datasets import load_dataset, split_dataset_kfold
from study.models.callbacks import get_callbacks_list

from util.constants import HRSTC
from util.data_types.trial_detail import TrialDetail


class KFoldTrial:
    """
    KFold validation (K = 5)
    Also provides some abstractions of model methods: fit(training), prediction
    """

    def __init__(self, trial_detail: TrialDetail):
        self.trial_detail = trial_detail

        self.ds_train, self.ds_test = load_dataset(self.trial_detail)

        self.epochs = HRSTC["EPOCHS_MAP"][trial_detail.dataset_name]
        self.all_histories = list()

    def fit(self):
        fold_datasets = split_dataset_kfold(self.ds_train, HRSTC["NUM_FOLDS"])

        for fold, (ds_train_fold, ds_val_fold) in enumerate(fold_datasets):
            model = build_model(self.trial_detail)

            if fold == 0:
                model.summary()

            callbacks = get_callbacks_list(self.trial_detail, fold)
            history = model.fit(
                ds_train_fold,
                epochs=self.epochs,
                validation_data=ds_val_fold,
                callbacks=callbacks,
            )

            self.all_histories.append(history)

        # Retrain on entire dataset
        self.model = build_model(self.trial_detail)

        callbacks = get_callbacks_list(self.trial_detail, 1)
        history = self.model.fit(
            self.ds_train,
            epochs=self.epochs,
            callbacks=callbacks,
        )

        return self.all_histories

    def predict(self):
        return self.model.evaluate(self.ds_test)
