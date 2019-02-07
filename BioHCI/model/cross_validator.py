import torch
from abc import ABC, abstractmethod
import numpy as np
import time
import BioHCI.helpers.utilities as utils
from tensorboardX import SummaryWriter

class CrossValidator(ABC):
	def __init__(self, subject_dict, data_splitter, dataset_processor, model, parameter, learning_def, all_categories):
		self._subject_dict = subject_dict
		self._data_splitter = data_splitter
		self._dataset_processor = dataset_processor
		self.__model = model
		self._learning_def = learning_def
		self._parameter = parameter
		self._num_folds = parameter.num_folds
		self._all_categories = all_categories
		self._cat_mapping = self.__map_categories(all_categories)

		self.all_int_categories = self.convert_categories(all_categories)

		self._all_val_accuracies = []
		self._all_train_accuracies = []
		self._all_train_losses = []

		# declare variables that will contain time needed to compute these operations
		self._cv_time = 0
		self._train_time = 0
		self._val_time = 0

		self.__tbx_path = utils.create_dir('tensorboardX_runs')
		self.__writer = SummaryWriter(self.__tbx_path)

		# create a confusion matrix to track correct guesses (accumulated over all folds of the Cross-Validation below)
		# TODO: oh noo!! there are two confusion matrixes - fix this - maybe use as a test case
		self._confusion = torch.zeros(len(all_categories), len(all_categories))

		self.perform_cross_validation()
		self._confusion_matrix = np.zeros((len(all_categories), len(all_categories)))

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

			# starting training with the above-defined parameters
			train_start = time.time()
			self.train(processed_train, self.__writer)
			self._train_time = utils.time_since(train_start)

			# start validating the model
			val_start = time.time()
			self.val(processed_val, self.__writer)
			self._val_time = utils.time_since(val_start)

		self._cv_time = utils.time_since(cv_start)

	@abstractmethod
	def _get_data_and_labels(self, python_dataset):
		pass

	@abstractmethod
	def train(self, train_dataset, summary_writer):
		pass

	@abstractmethod
	def val(self, val_dataset, summary_writer):
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
		for i in range(self._learning_def.num_epochs):
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
			subj_data = subj.get_data()
			subj_cat = subj.get_categories()

			for i, cat_data in enumerate(subj_data):
				for j in range(0, cat_data.shape[0]):
					chunk = cat_data[j, :, :]  # current chunk
					cat = subj_cat[i]  # current category - same within all the chunks of the innermost loop

					all_data.append(chunk)
					all_cat.append(cat)

		all_data = np.stack(all_data, axis=0)
		all_cat = np.array(all_cat)
		return all_data, all_cat

	def __map_categories(self, categories):
		"""
			Maps categories from a string element to an integer.

		Args:
			categories (list): List of unique string category names

		Returns:
			cat (dict): a dictionary mapping a sting to an integer

		"""
		# assert uniqueness of list elements
		assert len(categories) == len(set(categories))
		cat = {}

		for idx, elem in enumerate(categories):
			cat[elem] = idx

		return cat

	def convert_categories(self, categories):
		"""
		Converts a list of categories from strings to integers based on the internal attribute _cat_mapping.

		Args:
			categories (list): List of string category names of a dataset

		Returns:
			converted_categories (list): List of the corresponding integer id of the string categories

		"""
		converted_categories = []
		for idx, elem in enumerate(categories):
			assert elem in self._cat_mapping.keys()
			converted_categories.append(self._cat_mapping[elem])

		converted_categories = np.array(converted_categories)
		return converted_categories