"""
Created: 11/7/19
© Denisa Qori McDonald 2019 All Rights Reserved
"""
from abc import ABC, abstractmethod
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
