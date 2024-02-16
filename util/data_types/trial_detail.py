from dataclasses import dataclass
from textwrap import dedent

from util.constants import HRSTC


@dataclass
class TrialDetail:
    dataset_name: str
    augmentation_method: str
    percentage_data: float = 1
    energy_factor: float = HRSTC["MAX_ENERGY_FACTOR"]

    def __getitem__(self, idx):
        props = (
            self.dataset_name,
            self.augmentation_method,
            self.percentage_data,
            self.energy_factor,
        )
        return props[idx]

    def __repr__(self):
        energy_factor_repr = (
            f"{self.energy_factor:.2%}"
            if self.augmentation_method in {"all", "svd"}
            else "N/A"
        )
        return dedent(
            f"""
        Model & Dataset Name: {self.dataset_name.upper()}
        Percentage of Data Used: {self.percentage_data:.2%}
        Augmentation Method: {self.augmentation_method.upper()}
        Energy Factor: {energy_factor_repr}
        """
        )
