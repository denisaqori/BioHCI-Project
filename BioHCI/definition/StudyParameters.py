import os


# TODO: get rid of getters and setters? Can use Python's inherent getattr, setattr maybe - look into properties too
class StudyParameters:
	def __init__(self):
		# directory where the text data to be processed is found (all files in there are used)
		# 'Resources/fNIRS_boredom_data' correspond to boredom data
		self.__dir_path = 'Resources/EEG_workload_data'
		self.__full_dir_path = self.build_directory_path()

		self.__file_format = ".csv"

		# data information

		# these values represent column indices to retain from the data in each file
		# self.__relevant_columns = [1, 5, 9, 13, 17, 21]
		self.__relevant_columns = [0, 1, 2, 3]
		# this value represents the beginning of valid rows in the file
		self.__start_row = 0

		# the number of features found in each file (initial columns)
		self.__num_features = len(self.__relevant_columns)
		# the total number of subjects
		self.__num_subj = 12
		# whether the file for each subject contains labels in the last column (only!!)
		self.__labels_in = False
		# whether the dataset is to be standardized (default is per subject per column)
		self.__standardize = False

		# data processing information
		self.__samples_per_chunk = 30  # when chunking the dataset, the number of instances/samples in one chunk
		self.__interval_overlap = True  # determining whether to overlap instances while chunking

		self.__construct_features = False  # determining whether to construct features
		self.__feature_window = 10  # the number of measurements over which to define features (collapsing all to 1)

		self.__num_folds = 5  # The number of folds for cross-validation

		# run information
		self.__num_threads = 32  # The number of threads to be used during training (for gradient computing and
		# loading)
		# num_features = data.num_features  # number of features which determines input size
		self.__deep_learning = True

		# this variable determines whether training and testing happens within the same subject (therefore needing
		# calibration data to place data from any new subject), or is subject-independent (probably harder to get
		# higher accuracy)
		self.__calibration_free = True

		# the name of the study
		self.__study_name = "EEG_Workload"
		# the type of sensor data used - currently not being used anywhere but for bookkeeping
		self.__sensor_type = "EEG"

	# getters used by the program to obtain parameters
	def build_directory_path(self):
		return os.path.abspath(os.path.join(os.pardir, self.__dir_path))

	def get_dir_path(self):
		return self.__dir_path

	def get_full_dir_path(self):
		return self.__full_dir_path

	def get_file_format(self):
		return self.__file_format

	def get_relevant_columns(self):
		return self.__relevant_columns

	def get_start_row(self):
		return self.__start_row

	def get_num_subj(self):
		return self.__num_subj

	def get_num_features(self):
		return self.__num_features

	def is_labels_in(self):
		return self.__labels_in

	def is_standardize(self):
		return self.__standardize

	def get_num_threads(self):
		return self.__num_threads

	def is_deep_learning(self):
		return self.__deep_learning

	def is_construct_features(self):
		return self.__construct_features

	def get_feature_window(self):
		assert (self.__construct_features is True), "Features are not to be constructed in this configuration"
		return self.__feature_window

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

	# setters - for later use, especially if a UI gets built on top
	def set_dir_path(self, dir_path):
		self.__dir_path = dir_path

	def set_file_format(self, file_format):
		self.__file_format = file_format

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
