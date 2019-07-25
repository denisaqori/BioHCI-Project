from abc import ABC, abstractmethod


class LearningDefinition(ABC):
    def __init__(self, input_size):
        self.__input_size = input_size

    # self.__model = self._build_model(self.model_name)

    # @abstractmethod
    # def _build_model(self, name):
    # 	pass

    # @property
    # def learning(self):
    # 	return self.__model

    @property
    def input_size(self):
        return self.__input_size

    @property
    def batch_size(self):
        return 0