from abc import ABC
from typing import Optional


class LearningDefinition(ABC):

    def __init__(self, input_size):
        self.__input_size = input_size

    @property
    def num_hidden(self) -> Optional[int]:
        return None

    @property
    def num_epochs(self) -> Optional[int]:
        return None

    @property
    def batch_size(self) -> Optional[int]:
        return None

    @property
    def batch_first(self) -> Optional[bool]:
        return None

    @property
    def learning_rate(self) -> Optional[float]:
        return None

    @property
    def dropout_rate(self) -> Optional[float]:
        return None

    @property
    def num_layers(self) -> Optional[int]:
        return None

    @property
    def use_cuda(self) -> Optional[bool]:
        return None

    @property
    def input_size(self) -> Optional[int]:
        return None

    @property
    def output_size(self) -> Optional[int]:
        return None

    @property
    def nn_name(self) -> Optional[str]:
        return None
