import numpy as np
from BioHCI.data.data_constructor import AbstractData
from abc import ABC, abstractmethod


# This class uses as a label the neuroticism score achieved by each subject in the Big 5 Personality Test
class SubjectSpecificData(AbstractData, ABC):
	def __init__(self, parameter):
		print("Initializing a SubjectSpecificData class, which is a class to represent data where each subject"
			   "has an identifier or score that is not grouped in with other subjects (can be used for regression).")

		super(SubjectSpecificData, self).__init__(parameter)

	# information that needs to be implemented by each sub-class separately
	@abstractmethod
	def create_categories(self):
		pass

	# for each element of the subject list, find the filename it comes from, get its index in
	# the filename list, and then use that index to access the data of that particular file
	# the indices of filename_list and file_dataset match; ultimately append the data to index_labeled_dataset
	# so the data there matches by index to the category data		
	def index_dataset_by_label(self):
		print("Subject Directory List: ", self.subj_dir_list, "\n")

		for subj in self.subj_dir_list:
			index = self.subj_dir_list.index(str(subj))
			category_data = self.file_dataset[index]

			# add an extra first dimension to the numpy array, so it matches the format 
			# of GenderData, another AbstractData subclass. Shape changes 
			# from (inst_per_subj x num_attr) to (1 x inst_per_subj x num_attr)
			category_data = np.expand_dims(category_data, axis=0)

			print("Subject number: ", subj, "; Index in filename_list: ", index)
			self.index_labeled_dataset.append(category_data)


