import torch
from abc import ABC, abstractmethod
import numpy as np
import time
import BioHCI.utilities.Utilities as utils

class CrossValidation(ABC):
	def __init__(self, subject_dict, data_splitter, dataset_processor, parameter, learning_def, num_categories):
		self._subject_dict = subject_dict
		self._data_splitter = data_splitter
		self._dataset_processor = dataset_processor
		self._learning_def = learning_def
		self._parameter = parameter
		self._num_folds = parameter.get_num_folds()
		self._num_categories = num_categories

		self._all_val_accuracies = []
		self._all_train_accuracies = []
		self._all_train_losses = []

		# declare variables that will contain time needed to compute these operations
		self._cv_time = 0
		self._train_time = 0
		self._val_time = 0

		# create a confusion matrix to track correct guesses (accumulated over all folds of the Cross-Validation below)
		# TODO: oh noo!! there are two confusion matrixes - fix this - maybe use as a test case
		self._confusion = torch.zeros(num_categories, num_categories)

		self.perform_cross_validation()
		self._confusion_matrix = np.zeros((num_categories, num_categories))

	def perform_cross_validation(self):
		cv_start = time.time()

		for i in range(0, self._num_folds):
			print("\n\n"
			"*******************************************************************************************************")
			print("Run: ", i)
			train_dict, val_dict = self._data_splitter.split_into_folds(subject_dictionary=self._subject_dict,
																		num_folds=self._num_folds, val_index=i)

			processed_train = self._dataset_processor.process_dataset(train_dict)
			processed_val = self._dataset_processor.process_dataset(val_dict)

			print("Processed train dataset: ", processed_train)
			print("Processed val dataset: ", processed_val)

			# starting training with the above-defined parameters
			train_start = time.time()
			self.train(train_dataset)
			self._train_time = utils.time_since(train_start)

			# start validating the model
			val_start = time.time()
			self.val(val_dataset)
			self._val_time = utils.time_since(val_start)

		self._cv_time = utils.time_since(cv_start)

	@abstractmethod
	def _get_data_and_labels(self, python_dataset):
		pass

	@abstractmethod
	def train(self, train_dataset):
		pass

	@abstractmethod
	def val(self, val_dataset):
		pass

	def get_all_val_accuracies(self):
		return self._all_val_accuracies

	def get_all_train_accuracies(self):
		return self._all_train_accuracies

	# returns the average of train accuracy of each fold's last epoch
	def get_avg_train_accuracy(self):
		# return the average by dividing by the number of folds (=number of accuracies added)
		avg_accuracy = sum(self._all_train_accuracies) / float(len(self._all_train_accuracies))
		print("\nAverage train accuracy over", self._num_folds, "is", avg_accuracy)
		return avg_accuracy

	def get_avg_val_accuracy(self):
		avg_accuracy = sum(self._all_val_accuracies) / float(len(self._all_val_accuracies))
		print("\nAverage val accuracy over", self._num_folds, "is", avg_accuracy)
		return avg_accuracy

	def get_all_train_losses(self):
		return self._all_train_losses

	def get_avg_train_losses(self):
		avg_losses = []
		for i in range(self._learning_def.get_num_epochs()):
			epoch_loss = 0
			for j, loss_list in enumerate(self._all_train_losses):
				epoch_loss = epoch_loss + loss_list[i]
			avg_losses.append(epoch_loss / self._num_folds)
		return avg_losses

	def get_total_cv_time(self):
		return self._cv_time

	def get_train_time(self):
		return self._train_time

	def get_val_time(self):
		return self._val_time


