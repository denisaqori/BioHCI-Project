import os
import pprint as pp
import re
import sys

from scipy import stats

from BioHCI.data.subject import Subject


# TODO: add standardization option within feature for every subject - maybe entails building a dataframe first and
# then splitting again
class DataConstructor:
	def __init__(self, parameter):

		self.__parameter = parameter

		# number of files to be included in the dataset where each file contains fNIRS data for one subject
		self.__num_subj = parameter.num_subj

		# parameter's dir_path property determines the location of the data files, to be found within separate
		# directories for each subject. These are sub-directories of this path.
		self.__dir_path = parameter.dir_path

		self.subj_dir_list = self.get_subj_dir_list()
		assert self.__num_subj == len(self.subj_dir_list), "Not as many subject directories found as declared in " \
														   "StudyParameters"
		print("Number of total subjects to be included in the dataset: ", self.__num_subj)
		print("List of sub-directory names from where data will be obtained for each subject: ", self.subj_dir_list)

		self._subject_identity_list = self.create_subject_identity_list(self.subj_dir_list)
		print("\nSubject List: ", self._subject_identity_list, "\n")

		self.subj_dataset = self.build_all_subj_dataset(self.subj_dir_list)

		self.print_all_subj_dataset()


	# this method determines the path to the directory with each subject's data files and returns a list of names of
	# directories which should be of each subject
	def get_subj_dir_list(self):
		# we start by iterating through each .txt/.csv file in the given path
		directory = os.path.abspath(os.path.join(os.pardir, self.__dir_path))

		# we start a list where we can store the file names we are using,
		# since a lot of labeling depends on each file
		subj_dir_list = []

		for subj_dir in os.listdir(directory):
			dir_name = os.fsdecode(subj_dir)
			subj_dir_list.append(dir_name)

		return subj_dir_list

	# this method returns a Python dictionary of Subjects, with the key being the subject number and value being the
	# Subject object. Each Subject contains his/her own data split by categories
	# (one or more). The data itself can be accessed by the calling the Subject class methods.
	def build_all_subj_dataset(self, subj_dir_list):

		all_subj = {}

		# for each subject directory, create the whole path and give that to the Subject class
		# in order for it to build the dataset from files found there
		for subj_dir_name in subj_dir_list:
			subj_data_path = os.path.join(self.__dir_path, subj_dir_name)

			subj = Subject(subj_data_path, self.__parameter)
			all_subj[subj_dir_name] = subj

		return all_subj

	# print and plot information about the subject dataset -  subject names and data shapes per category
	def print_all_subj_dataset(self):

		print("\nSubject dataset:")
		# for subj_dir_name in self.subj_dir_list:
		# 	subject = self.subj_dataset[subj_dir_name]
		for subj_name, subj in self.subj_dataset.items():
			print("Subj", subj_name, "- shapes of each category data with the corresponding categories:")

			for i, cat_data in enumerate(subj.get_data()):
				pp.pprint(cat_data.shape)

			pp.pprint(subj.get_categories())
			print("\n")

	# create a subject identity list, where each subject is an integer, and the list is sorted
	def create_subject_identity_list(self, subj_dir_list):
		subject_list = []
		for subj_dir_name in subj_dir_list:
			# remove non-number characters from directory name
			subj_dir_name = re.sub("\D", "", subj_dir_name)
			try:
				subj = int(subj_dir_name)
			except ValueError:
				print("Sub-directories must be named by the number of the subject only")
				print("Exiting...")
				sys.exit()
			subject_list.append(subj)

		subject_list.sort()
		return subject_list

	# this method standardizes the dataset that is passed to it. The format of the dataset variable is assumed to be
	# the same as that of index_labeled_dataset: a python list of numpy arrays of shape
	# ((number of subjects in category) x (number of instances per file/subject) x (number of features))
	# TODO: needs to change to fit the new structure of the dataframe
	def standardize(self, dataset, std_type):
		standardized_dataset = []
		if std_type is 'PerSubjPerColumn':
			print("\nCreating a standardized (z-score) dataset by subject and channel...\n")
			for j, category_data in enumerate(dataset):
				# standardize across the second axis (inst_per_subj)
				standardized_category = stats.zscore(category_data, axis=1)
				standardized_dataset.append(standardized_category)
		elif std_type is 'perSubj':
			print("Creating a standardized (z-score) dataset by subject only...\n")
			for j, category_data in enumerate(dataset):
				# standardize the whole array since no axis is specified
				standardized_category = stats.zscore(category_data)
				standardized_dataset.append(standardized_category)
		else:
			print("Standardization method defined in AbstractData not implemented!")
			print("Exiting...")
			sys.exit()

		return standardized_dataset

	def get_subject_dataset(self):
		return self.subj_dataset

	# this function returns the set of all categories found across all subjects in the study (all unique values)
	def get_all_dataset_categories(self):
		category_list = []
		for subj_name, subj in self.subj_dataset.items():
			category_list.append(subj.get_categories())

		flat_list = [item for sublist in category_list for item in sublist]
		categories = list(set(flat_list))

		# return the unique values in flat_list
		return categories

