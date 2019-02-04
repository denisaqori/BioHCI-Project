from abc import ABC, abstractmethod


class LearningDefinition(ABC):
	def __init__(self, model_name):
		self.__model_name = model_name
		self.__model = self._build_model(self.model_name)

	@abstractmethod
	def _build_model(self, name):
		pass

	@property
	def model(self):
		return self.__model

	@property
	def model_name(self):
		return self.__model_name
