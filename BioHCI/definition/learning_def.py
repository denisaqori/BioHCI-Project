from abc import ABC, abstractmethod


class LearningDefinition(ABC):
	def __init__(self, model_name):
		self.model_name = model_name
		self.model = self._build_model(self.model_name)

	@abstractmethod
	def _build_model(self, name):
		pass

	def get_model(self):
		return self.model

	def get_model_name(self):
		return self.model_name
