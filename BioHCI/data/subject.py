import os
import numpy as np
import re


class Subject:
	def __init__(self, subj_data_path, parameter):
		self.__subj_data_path = subj_data_path
		self.__parameter = parameter

		self.__filename_list = []

		self.__data, self.__categories = self.__build_subj_data()

		assert len(self.__data) == len(self.__categories), "The sizes of the subject's data list and categories list " \
														   "do not match!!"
		self.__all_data_bool = True

	# this method returns a python list of numpy arrays with all the signal data from the text files of one subject
	# for each category/label. The data for each category is expected to be found in the subject sub-directory
	# in a separate file and is represented as a numpy array, element of the returned python list
	def __build_subj_data(self):
		print("\nBuilding the subject dataset: ")

		# get all the files where this subject's data is found

		subj_category_data = []
		subj_category_names = []
		for cat_data_container in os.listdir(self.__subj_data_path):
			# each subject should have a directory for each category
			subj_cat_data_path = os.path.join(self.__subj_data_path, cat_data_container)
			if os.path.isdir(subj_cat_data_path):

				for filename in os.listdir(subj_cat_data_path):
					if filename.endswith((self.__parameter.file_format)):

						filepath = os.path.join(subj_cat_data_path, filename)
						filedata = self.__get_file_data(filepath)

						subj_category_data.append(filedata)
						subj_category_names.append(cat_data_container)

		return subj_category_data, subj_category_names

	def __get_file_data(self, filepath):
		with open(filepath, encoding='ascii') as f:

			# get the data in each file by first stripping and splitting the lines and
			# then creating a numpy array out of these values
			file_lines = []
			print("Filename: ", filepath)
			for line in f:
				line = line.strip(' \t\n\r')
				line = re.split('\t|,', line)
				file_lines.append(line)
			file_lines = np.asarray(file_lines)

			# keep info only from the relevant columns and rows
			file_lines = (file_lines[self.__parameter.start_row:,
						  self.__parameter.relevant_columns]).astype(np.float32)
			# subj_category_data.append(file_lines)
			return file_lines

	# this method creates categories of the subject; if labels_in in parameter is set to True, the categories are
	# acquired from the dataset, otherwise from the file names within the subject directory
	# def __create_subject_categories(self):
	# 	categories = []
	# 	print("labels_col: ", type(self.__parameter.labels_col))
	# 	if self.__parameter.labels_col is None:
	# 		for filename in self.__filename_list:
	# 			keep the filename only to assign the category, and remove the file extension (format)
				# category_name = filename[:-len(self.__parameter.file_format)]
				# categories.append(category_name)
		# return categories

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
