import os
import numpy as np
import re


class Subject:
	def __init__(self, subj_data_path, parameter):
		self.__subj_data_path = subj_data_path
		self.__parameter = parameter

		self.__filename_list = []

		self.__data = self.__build_subj_data()
		self.__categories = self.__create_subject_categories()

		assert len(self.__data) == len(self.__categories), "The sizes of the subject's data list and categories list " \
														   "do not match!!"
		self.__all_data_bool = True

	# this method returns a python list of numpy arrays with all the signal data from the text files of one subject
	# for each category/label. The data for each category is expected to be found in the subject sub-directory
	# in a separate file and is represented as a numpy array, element of the returned python list
	def __build_subj_data(self):
		print("\nBuilding the subject dataset: ")

		# get all the files where this subject's data is found

		for filename in os.listdir(self.__subj_data_path):
			if filename.endswith((self.__parameter.get_file_format())):
				self.__filename_list.append(filename)

		subj_category_data = []

		i = 0
		for filename in self.__filename_list:
			full_path = os.path.join(self.__subj_data_path, filename)
			with open(full_path, encoding='ascii') as f:

				# get the data in each file by first stripping and splitting the lines and
				# then creating a numpy array out of these values
				file_lines = []
				print("Filename: ", full_path)
				for line in f:
					line = line.strip(' \t\n\r')
					line = re.split('\t|,', line)
					file_lines.append(line)
				file_lines = np.asarray(file_lines)

				# keep info only from the relevant columns and rows
				file_lines = (file_lines[self.__parameter.get_start_row():,
							  self.__parameter.get_relevant_columns()]).astype(np.float32)
				subj_category_data.append(file_lines)
				i = i + 1

		return subj_category_data

	# this method creates categories of the subject; if labels_in in parameter is set to True, the categories are
	# acquired from the dataset, otherwise from the file names within the subject directory
	def __create_subject_categories(self):
		categories = []
		if not self.__parameter.is_labels_in():
			for filename in self.__filename_list:
				# keep the filename only to assign the category, and remove the file extension (format)
				category_name = filename[:-len(self.__parameter.get_file_format())]
				categories.append(category_name)
		return categories

	# return subject data split by categories, as a list of numpy arrays
	def get_data(self):
		return self.__data

	# returns subject categories
	def get_categories(self):
		return self.__categories

	def set_data(self, data):
		self.__data = data

	def set_categories(self, categories):
		self.__categories = categories

	def get_all_data_bool(self):
		return self.__all_data_bool

	def set_all_data_bool(self, all_data_bool):
		self.__all_data_bool = all_data_bool
