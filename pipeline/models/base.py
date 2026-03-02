"""Abstract base class for all model implementations."""
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from torch_geometric.data import HeteroData


class BaseModel(ABC):
    def __init__(
        self,
        cfg: dict,
        data: HeteroData,
        drug_map: dict,
        disease_map: dict,
    ):
        self.cfg = cfg
        self.data = data
        self.drug_map = drug_map
        self.disease_map = disease_map

    @abstractmethod
    def train_and_eval(
        self,
        train_idx: np.ndarray,
        test_idx: np.ndarray,
        samples_df: pd.DataFrame,
        drug_local: np.ndarray,
        dis_local: np.ndarray,
        y: np.ndarray,
        label: str = '',
    ) -> dict:
        """Train on train_idx pairs, evaluate on test_idx pairs.

        Returns
        -------
        dict with keys: auc, acc, f1
        """
        ...
