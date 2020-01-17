"""
Created: 11/7/19
© Denisa Qori McDonald 2019 All Rights Reserved
"""
from abc import ABC, abstractmethod
from typing import Optional

import numpy as np


class KnittedComponent(ABC):
    def __init__(self):
        super(KnittedComponent, self).__init__()

    @property
    @abstractmethod
    def name(self):
        pass

    @abstractmethod
    def get_button_centers(self, ids: np.ndarray):
        pass

    def get_row_labels(self, cat):
        pass

    @property
    def num_rows(self) -> Optional[int]:
        return None

    @property
    def num_cols(self) -> Optional[int]:
        return None

    @property
    def num_buttons(self) -> Optional[int]:
        return None
