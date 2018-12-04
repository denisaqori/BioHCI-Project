"""
© Denisa Qori McDonald 2018 All Rights Reserved
"""

import os
from BioHCI.helpers import utilities as util


class StudyParameters:
	def __init__(self):
		# directory where the text data to be processed is found (all files in there are used)
		# 'Resources/fNIRS_boredom_data' correspond to boredom data
		self.__dir_path = None

		self.__file_format = None

		# data information

		# these values represent column indices to retain from the data in each file
		# self.__relevant_columns = [1, 5, 9, 13, 17, 21]
		self.__relevant_columns = None
		# this value represents the beginning of valid rows in the file
		self.__start_row = None

		# the total number of subjects
		self.__num_subj = None
		# the column in which the file for each subject contains labels. If set to None, there are no labels.
		self.__labels_col = None

		# data processing information

		# whether the dataset is to be standardized (default is per subject per column)
		self.__standardize = None

		self.__samples_per_chunk = None  # when chunking the dataset, the number of instances/samples in one chunk
		self.__interval_overlap = None  # determining whether to overlap instances while chunking

		self.__construct_features = None  # determining whether to construct features
		self.__feature_window = None  # the number of measurements over which to define features (collapsing all to 1)

		self.__num_folds = None  # The number of folds for cross-validation

		# run information
		self.__num_threads = None  # The number of threads to be used during training (for gradient computing and
		# loading)
		# num_features = data.num_features  # number of features which determines input size
		self.__neural_net = None

		# this variable determines whether training and testing happens within the same subject (therefore needing
		# calibration data to place data from any new subject), or is subject-independent (probably harder to get
		# higher accuracy)
		self.__calibration_free = None

		# the name of the study
		self.__study_name = None
		# the type of sensor data used - currently not being used anywhere but for bookkeeping
		self.__sensor_name = None

	@property
	def dir_path(self):
		return self.__dir_path

	@dir_path.setter
	def dir_path(self, dir_path):
		project_root_path = util.get_root_path("Resources")
		path = os.path.abspath(os.path.join(project_root_path, dir_path))

		assert (os.path.exists(path)), "The directory \'" + path + "\' does not exist. Ensure the dataset is " \
																   "properly placed."
		self.__dir_path = path

	@property
	def file_format(self):
		return self.__file_format

	@file_format.setter
	def file_format(self, file_format):
		assert file_format.startswith('.'), "The file ending name should start with a '.'"
		self.__file_format = file_format

	@property
	def relevant_columns(self):
		return self.__relevant_columns

	@relevant_columns.setter
	def relevant_columns(self, relevant_columns):
		assert (isinstance(relevant_columns, list)), "The relevant columns to process need to be passed in a list."
		for elem in relevant_columns:
			assert (isinstance(elem, int) and int(elem) >= 0), "Each element of the list needs to be a positive " \
															   "integer or zero."
		self.__relevant_columns = relevant_columns

	@property
	def start_row(self):
		return self.__start_row

	@start_row.setter
	def start_row(self, start_row):
		assert (isinstance(start_row, int) and (
				int(start_row) >= 0)), "Number of subjects needs to be a positive integer or zero."
		self.__start_row = start_row

	@property
	def num_subj(self):
		return self.__num_subj

	@num_subj.setter
	def num_subj(self, num_subj):
		assert (isinstance(num_subj, int) and (int(num_subj) > 0)), "Number of subjects needs to be a positive integer."
		self.__num_subj = num_subj

	@property
	def num_features(self):
		return len(self.__relevant_columns)

	@property
	def labels_col(self):
		if self.__labels_col is "None":
			self.__labels_col = None
		return self.__labels_col

	@labels_col.setter
	def labels_col(self, labels_col):
		if labels_col is not "None":
			assert (isinstance(labels_col, int) and int(labels_col)) >= 0, "Label column should be 0 or a positive " \
																		   "integer if labels are included with the " \
																		   "data, and \"None\" otherwise."
		self.__labels_col = labels_col

	@property
	def standardize(self):
		return self.__standardize

	@standardize.setter
	def standardize(self, standardize):
		assert (isinstance(standardize, bool)), "The standardize variable needs to be a boolean to indicate " \
												"whether the dataset is to be standardized."
		self.__standardize = standardize

	@property
	def num_threads(self):
		return self.__num_threads

	@num_threads.setter
	def num_threads(self, num_threads):
		assert (isinstance(num_threads, int) and (num_threads > 0)), "Number of threads needs to be a positive integer."
		self.__num_threads = num_threads

	@property
	def neural_net(self):
		return self.__neural_net

	@neural_net.setter
	def neural_net(self, neural_net):
		assert (isinstance(neural_net, bool)), "The neural_net variable needs to be a boolean to indicate " \
											   "whether the model uses a neural network-based network. "
		self.__neural_net = neural_net

	@property
	def calibration_free(self):
		return self.__calibration_free

	@calibration_free.setter
	def calibration_free(self, calibration_free):
		assert (isinstance(calibration_free, bool)), "The variable calibration_free needs to be a boolean."
		self.__calibration_free = calibration_free

	@property
	def construct_features(self):
		return self.__construct_features

	@construct_features.setter
	def construct_features(self, construct_features):
		assert (isinstance(construct_features, bool)), "The construct_feature variable needs to be of type boolean."
		self.__construct_features = construct_features

	@property
	def feature_window(self):
		assert (self.__construct_features is True), "Features are not to be constructed in this configuration"
		return self.__feature_window

	@feature_window.setter
	def feature_window(self, feature_window):
		assert (isinstance(feature_window, int) and (int(feature_window) > 0)), "Feature window needs to be a " \
																				"positive integer."
		self.__feature_window = feature_window

	@property
	def samples_per_chunk(self):
		return self.__samples_per_chunk

	@samples_per_chunk.setter
	def samples_per_chunk(self, samples_per_chunk):
		assert (isinstance(samples_per_chunk, int) and (samples_per_chunk > 0)), "Samples per chunk needs to be a " \
																				 "positive integer."
		self.__samples_per_chunk = samples_per_chunk

	@property
	def interval_overlap(self):
		return self.__interval_overlap

	@interval_overlap.setter
	def interval_overlap(self, interval_overlap):
		assert (isinstance(interval_overlap, bool)), "Interval overalap needs to be a boolean, showing whether when " \
													 "chunking the dataset and creating intervals, intermediate " \
													 "intervals consisting of 50% of each are to be built."
		self.__interval_overlap = interval_overlap

	@property
	def num_folds(self):
		return self.__num_folds

	@num_folds.setter
	def num_folds(self, num_folds):
		assert (isinstance(num_folds, int) and (int(num_folds) > 0)), "Number of folds for cross validation needs to " \
																	  "be a positive integer."
		self.__num_folds = num_folds

	@property
	def study_name(self):
		return self.__study_name

	@study_name.setter
	def study_name(self, study_name):
		self.__study_name = study_name

	@property
	def sensor_name(self):
		return self.__sensor_name

	@sensor_name.setter
	def sensor_name(self, sensor_name):
		self.__sensor_name = sensor_name

	def clear_attribute_values(self):
		"""
		Sets each attribute of the sole StudyParameters instance object in use to 'None'.

		"""
		print ("Setting every attribute in the sole StudyParameter instance to 'None'.")
		for attr, val in vars(self).items():
			self.__setattr__(attr, "None")
		print (vars(self))
