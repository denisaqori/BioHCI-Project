'''
Created: 11/30/18
© Denisa Qori McDonald 2018 All Rights Reserved
'''

import os
from BioHCI.helpers import utilities as util


# TODO: get rid of getters and setters? Can use Python's inherent getattr, setattr maybe - look into properties too
class StudyParameters:
	def __init__(self):
		# directory where the text data to be processed is found (all files in there are used)
		# 'Resources/fNIRS_boredom_data' correspond to boredom data
		self.dir_path = 'EEG_workload_data'

		self.file_format = ".csv"

		# data information

		# these values represent column indices to retain from the data in each file
		# self.__relevant_columns = [1, 5, 9, 13, 17, 21]
		self.relevant_columns = [0, 1, 2, 3]
		# this value represents the beginning of valid rows in the file
		self.start_row = 0

		# the total number of subjects
		self.num_subj = 12
		# the column in which the file for each subject contains labels. If set to None, there are no labels.
		self.labels_col = None

		# data processing information

		# whether the dataset is to be standardized (default is per subject per column)
		self.standardize = False

		self.__samples_per_chunk = 30  # when chunking the dataset, the number of instances/samples in one chunk
		self.__interval_overlap = True  # determining whether to overlap instances while chunking

		self.__construct_features = False  # determining whether to construct features
		self.__feature_window = 10  # the number of measurements over which to define features (collapsing all to 1)

		self.__num_folds = 5  # The number of folds for cross-validation

		# run information
		self.__num_threads = 32  # The number of threads to be used during training (for gradient computing and
		# loading)
		# num_features = data.num_features  # number of features which determines input size
		self.neural_net = True

		# this variable determines whether training and testing happens within the same subject (therefore needing
		# calibration data to place data from any new subject), or is subject-independent (probably harder to get
		# higher accuracy)
		self.__calibration_free = True

		# the name of the study
		self.__study_name = "EEG_Workload"
		# the type of sensor data used - currently not being used anywhere but for bookkeeping
		self.__sensor_type = "EEG"

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
		self.__file_format = file_format

	@property
	def relevant_columns(self):
		return self.__relevant_columns

	@relevant_columns.setter
	def relevant_columns(self, relevant_columns):
		assert (isinstance(relevant_columns, list)), "The relevant columns to process need to be passed in a list."
		self.__relevant_columns = relevant_columns

	@property
	def start_row(self):
		return self.__start_row

	@start_row.setter
	def start_row(self, start_row):
		assert (isinstance(start_row, int) and (int(start_row) >= 0)), "Number of subjects needs to be a positive " \
																	   "integer or zero."
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
		return self.__labels_col

	@labels_col.setter
	def labels_col(self, labels_col):
		if labels_col is not None:
			assert (isinstance(labels_col, int)), "The label column should be an integer."
			assert int(labels_col) >= 0, "Label column should be 0 or a positive integer if labels are " \
										 "included with the data, and None otherwise."
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
		assert (isinstance(num_threads, int) and (num_threads > 0)), "Number of threads needs to be a positive " \
																	 "integer."
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
		assert (isinstance(feature_window, int) and (feature_window > 0)), "Feature window needs to be a positive " \
																		   "integer."
		self.__feature_window = feature_window

	def is_interval_overlap(self):
		return self.__interval_overlap

	def get_samples_per_chunk(self):
		return self.__samples_per_chunk

	def get_num_folds(self):
		return self.__num_folds

	def get_study_name(self):
		return self.__study_name

	def get_sensor_type(self):
		return self.__sensor_type


'''
	# setters - for later use, especially if a UI gets built on top

	def set_relevant_columns(self, relevant_columns):
		self.__relevant_columns = relevant_columns

	def set_start_row(self, start_row):
		self.__start_row = start_row

	def set_num_subj(self, num_subj):
		self.__num_subj = num_subj

	def set_num_features(self, num_features):
		self.__num_features = num_features

	def set_bool_labels_in(self, labels_in):
		self.__labels_in = labels_in

	def set_bool_standardize(self, standardize):
		self.__standardize = standardize

	def set_num_threads(self, num_threads):
		self.__num_threads = num_threads

	def set_bool_deep_learning(self, deep_learning):
		self.__deep_learning = deep_learning

	def set_bool_construct_features(self, construct_features):
		self.__construct_features = construct_features

	def set_feature_window(self, feature_window):
		assert (self.__construct_features is True), "Features are not to be constructed in this configuration"
		assert isinstance(feature_window, int)
		self.__feature_window = feature_window

	def set_interval_overlap(self, interval_overlap):
		self.__interval_overlap = interval_overlap

	def set_num_folds(self, num_folds):
		self.__num_folds = num_folds

	def set_samples_per_chunk(self, samples_per_chunk):
		self.__samples_per_chunk = samples_per_chunk

	def set_study_name(self, study_name):
		self.__study_name = study_name

	def set_sensor_type(self, sensor_type):
		self.__sensor_type = sensor_type
'''
