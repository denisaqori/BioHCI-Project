from BioHCI.data.data_splitter import DataSplitter
from copy import copy
import math
import numpy as np


# This class includes data from each subject in the testing, training, and validation sets.
class WithinSubjectSplitter(DataSplitter):
	def __init__(self, subject_dictionary, train_val_percent=0.8, test_percent=0.2):
		super(WithinSubjectSplitter, self).__init__(subject_dictionary, train_val_percent, test_percent)

	def split_dataset(self, subject_dict, split_percent):
		""" Splits the dictionary passed as an argument into two sets, one to be used for training and validation,
			the other for testing.

			Each of the resulting dictionaries (train and testing) contain data from each subject. Useful for
			situations when qualities associated with each subject are to be predicted, or the subject's identity. For
			subject state prediction, which can change relatively quickly for the same subject,
			use AcrossSubjectSplitter.

		Args:
			subject_dict (dict): a dictionary mapping subject name to the corresponding Subject object. The Subject
			object data contains a list of 2D ndarrays (num total instances x num features).
			split_percent (float): the percentage of data from each category from the data of the Subject object to
				be included in the returned trained_dict, with the rest of the data returned in the eval_dict

		Returns:
			train_val_dict (dict): data to be used for training and validation. A dictionary mapping subject name to
				its Subject object. All subject names from the input dictionary (keys), have been appended the string
								'_train_val'.
			test_dict (dict): Data to be used for final evaluation. A dictionary mapping subject name to its Subject
				object.	All subject names from the input dictionary (keys), have been appended the string '_eval'.

		"""
		# create empty dictionaries to return
		train_val_dict = {}
		test_dict = {}

		# iterate over subject dictionary to get the data and corresponding categories per subject
		for subj_name, subject in subject_dict.items():
			subj_data = subject.data
			subj_cat = subject.categories

			# for each category get the first train_percent part to assign to training, and assign the rest to
			# testing
			subj_train_val_list = []
			subj_test_list = []
			for i, category in enumerate(subj_data):
				# get the first train_percent * total number of instances for each category for each subject to be
				# assigned to training and validation, and assign the remaining part for testing
				train_val_line_end = math.floor(category.shape[0] * split_percent)
				cat_train_val_data = category[0: train_val_line_end, :]
				subj_train_val_list.append(cat_train_val_data)

				cat_test_data = category[train_val_line_end:, :]
				subj_test_list.append(cat_test_data)

			# assign the data from the original subject to the two dictionaries to be returned

			# python copies by reference. If you do obj2 = obj1, obj2 will have the reference to obj1.
			# using copy as in: obj2 = copy(obj1), a new obj2 will be created with identical values as obj1,
			# but in a different memory location
			subj_train_val = copy(subject)
			subj_train_val.data = subj_train_val_list
			subj_train_val.categories = subj_cat
			subj_train_val.all_data_bool = False

			train_val_dict[subj_name + '_train_val'] = subj_train_val

			subj_test = copy(subject)
			subj_test.data = subj_test_list
			subj_test.categories = subj_cat
			subj_test.all_data_bool = False

			test_dict[subj_name + '_test'] = subj_test

		return train_val_dict, test_dict

	def split_into_folds(self, subject_dictionary, num_folds, val_index):
		"""
		Splits the data from each category form each subject into folds to be used for cross validation.

		Args:
			subject_dictionary (dict): a dictionary mapping a subject name to a subject object
			num_folds (int): the number of folds to split the dataset into
			val_index (int): the fold to be used for validation, with all the rest being used for training

		Returns:
			train_dict (dict): a dictionary mapping a subject name to a subject object, whose data is to be used for
				training. Each category's data for each subject is split into num_folds, and each subject in this
				dictionary contains num_folds - 1 of those for each category.
			val_dict (dict): a dictionary mapping a subject name to a subject object, whose data is to be used for
				validation. Each category's data for each subject is split into num_folds, and each subject in this
				dictionary contains 1 of those for each category.

		"""
		assert isinstance(num_folds, int), "num_folds needs to be an integer"
		assert isinstance(val_index, int), "val_index needs to be an integer"
		assert val_index < num_folds, "Not enough folds to index with val_index"

		train_dict = {}
		val_dict = {}
		for subj_name, subject in subject_dictionary.items():
			subj_data = subject.data

			train_list = []
			val_list = []
			for i, category in enumerate(subj_data):
				num_inst = category.shape[0]
				assert (num_folds < num_inst), "Number of folds to split the data into should be smaller " \
											   "than the first dimension of the " \
											   "subject's categories (number of instances)"

				# split the current category of the current subject across axis 0 into num_folds equal parts. The
				# parts need not be exactly equal, a some have an extra instance than the rest
				# look up numpy's array_split() for more information
				folds_list = np.array_split(category, num_folds, axis=0)
				val_list.append(folds_list.pop(val_index))

				# concatenate the remaining folds and append them to the train list
				train_array = folds_list.pop(0)
				for idx, fold in enumerate(folds_list):
					train_array = np.concatenate((train_array, fold), axis=0)
				train_list.append(train_array)

			# populate the train and validation dictionaries with the data for each category
			val_subj = copy(subject)
			val_subj.data = val_list
			val_subj.all_data_bool = False
			val_dict[subj_name.replace("_train", "")] = val_subj

			train_subj = copy(subject)
			train_subj.data = train_list
			train_subj.all_data_bool = False
			train_dict[subj_name.replace("_val", "")] = train_subj

		return train_dict, val_dict
