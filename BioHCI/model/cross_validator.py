import torch
from abc import ABC, abstractmethod
import numpy as np
import time
import BioHCI.helpers.utilities as utils
from tensorboardX import SummaryWriter
import BioHCI.helpers.type_aliases as types
from typing import Optional, List

from BioHCI.data.data_splitter import DataSplitter
from BioHCI.data_processing.feature_constructor import FeatureConstructor
from BioHCI.definition.learning_def import LearningDefinition
from BioHCI.definition.study_parameters import StudyParameters


class CrossValidator(ABC):
	def __init__(self, subject_dict: types.subj_dataset, data_splitter: DataSplitter, feature_constructor:
	FeatureConstructor, model, parameters: StudyParameters, learning_def: LearningDefinition, all_categories: List[str]):
		self.__subject_dict = subject_dict
		self.__data_splitter = data_splitter
		self.__feature_constructor = feature_constructor
		self.__model = model
		self.__learning_def = learning_def
		self.__parameters = parameters
		self.__num_folds = parameters.num_folds

		self.__all_int_categories = utils.convert_categories(all_categories, all_categories)

		self.__all_val_accuracies = []
		self.__all_train_accuracies = []
		self.__all_train_losses = []

		# declare variables that will contain time needed to compute these operations
		self.__cv_time = ""
		self.__train_time = ""
		self.__val_time = 0

		self.__tbx_path = utils.create_dir('tensorboardX_runs')
		self.__writer = SummaryWriter(self.tbx_path)

		# create a confusion matrix to track correct guesses (accumulated over all folds of the Cross-Validation
		# below)
		# TODO: oh noo!! there are two confusion matrixes - fix this - maybe use as a test case
		self._confusion = torch.zeros(len(all_categories), len(all_categories))

		self.perform_cross_validation()
		self._confusion_matrix = np.zeros((len(all_categories), len(all_categories)))

	@property
	def subject_dict(self) -> types.subj_dataset:
		return self.__subject_dict

	@property
	def data_splitter(self) -> DataSplitter:
		return self.__data_splitter

	@property
	def feature_constructor(self) -> FeatureConstructor:
		return self.__feature_constructor

	@property
	def model(self):
		return self.__model

	@property
	def learning_def(self) -> LearningDefinition:
		return self.__learning_def

	@property
	def parameters(self) -> StudyParameters:
		return self.__parameters

	@property
	def num_folds(self) -> int:
		return self.__num_folds

	@property
	def all_int_categories(self) -> np.ndarray:
		return self.__all_int_categories

	@property
	def all_val_accuracies(self) -> List[float]:
		return self.__all_val_accuracies

	@property
	def all_train_accuracies(self) -> List[float]:
		return self.__all_train_accuracies

	@property
	def all_train_losses(self) -> List[float]:
		return self.__all_train_losses

	@property
	def cv_time(self) -> str:
		return self.__cv_time

	@cv_time.setter
	def cv_time(self, time: str):
		self.__cv_time = time

	@property
	def train_time(self) -> str:
		return self.__train_time

	@train_time.setter
	def train_time(self, time: str):
		self.__train_time = time

	@property
	def val_time(self):
		return self.__val_time

	@val_time.setter
	def val_time(self, time: str):
		self.__val_time = time

	@property
	def tbx_path(self):
		return self.__tbx_path

	@property
	def writer(self):
		return self.__writer

	def perform_cross_validation(self) -> None:
		cv_start = time.time()

		for i in range(0, self.num_folds):
			print("\n\n"
			"*******************************************************************************************************")
			print("Run: ", i)
			train_dict, val_dict = self.data_splitter.split_into_folds(subject_dictionary=self.subject_dict,
																		num_folds=self.num_folds, val_index=i)

			processed_train = self.feature_constructor.produce_feature_dataset(train_dict)
			processed_val = self.feature_constructor.produce_feature_dataset(val_dict)

			# starting training with the above-defined parameters
			train_start = time.time()
			self.train(processed_train, self.writer)
			self.train_time = utils.time_since(train_start)

			# start validating the model
			val_start = time.time()
			self.val(processed_val, self.writer)
			self.val_time = utils.time_since(val_start)

		self.cv_time = utils.time_since(cv_start)

	@abstractmethod
	def _get_data_and_labels(self, python_dataset):
		pass

	@abstractmethod
	def train(self, train_dataset, summary_writer):
		pass

	@abstractmethod
	def val(self, val_dataset, summary_writer):
		pass

	@property
	def avg_train_accuracy(self) -> float:
		"Compute the average of train accuracy of each fold's last epoch."

		# return the average by dividing the sum by the number of folds (= number of accuracies added)
		avg_accuracy = sum(self.all_train_accuracies) / float(len(self.all_train_accuracies))
		print("\nAverage train accuracy over", self.num_folds, "is", avg_accuracy)
		return avg_accuracy

	@property
	def avg_val_accuracy(self) -> float:
		avg_accuracy = sum(self.all_val_accuracies) / float(len(self.all_val_accuracies))
		print("\nAverage val accuracy over", self.num_folds, "is", avg_accuracy)
		return avg_accuracy

	@property
	def avg_train_losses(self) -> List[float]:
		avg_losses = []
		for i in range(self.learning_def.num_epochs):
			epoch_loss = 0
			for j, loss_list in enumerate(self.all_train_losses):
				epoch_loss = epoch_loss + loss_list[i]
			avg_losses.append(epoch_loss / self.num_folds)
		return avg_losses

	def mix_subj_chunks(self, subj_dict):
		"""
		Creates a dataset of chunks of all subjects with the corresponding categories. At this point the subject data
		is not separated anymore.

		Args:
			subj_dict (dict): a dictionary mapping a subject name to a Subject object

		Returns:
			all_data (ndarray): a 3D numpy array containing the train dataset of shape (number of chunks x number of
				instances per chunk x number of features)
			all_cat (ndarray): a 1D numpy arrray containing the category labels of all_data, of shape (number of
				chunks).
		"""

		# data to stack - subjects end up mixed together in the ultimate dataset
		all_data = []
		# list of all categories to return
		all_cat = []

		for subj_name, subj in subj_dict.items():
			subj_data = subj.data
			subj_cat = subj.categories

			for i, cat_data in enumerate(subj_data):
				for j in range(0, cat_data.shape[0]):
					chunk = cat_data[j, :, :]  # current chunk
					cat = subj_cat[i]  # current category - same within all the chunks of the innermost loop

					all_data.append(chunk)
					all_cat.append(cat)

		all_data = np.stack(all_data, axis=0)
		all_cat = np.array(all_cat)
		return all_data, all_cat

