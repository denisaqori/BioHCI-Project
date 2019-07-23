from BioHCI.data.data_constructor import AbstractData
import numpy as np


class Boredom_GenderData(AbstractData):
	def __init__(self, parameter):
		super(Boredom_GenderData, self).__init__(parameter)

	def get_dataset_name(self):
		return "Gender Data"

	# information obtained from file BOYER_ALL_DATA.xlsx
	def create_categories(self):
		print("\nGender Data Processing object is being initialized...")
		self.__num_males = 12
		self.__num_females = 18
		self.categories.append('M')
		self.categories.append('F')

	def index_dataset_by_label(self):

		print("Creating the dataset by stacking subject data of the same label in the same numpy array.")

		# create numpy arrays for each class, to be populated by the file information below
		males_dataset = np.zeros(shape=(self.__num_males, self.get_inst_per_subject(), self.num_features))
		females_dataset = np.zeros(shape=(self.__num_females, self.get_inst_per_subject(), self.num_features))

		i = 0
		j = 0
		for filename in self.filename_list:

			# for the given filename, find the index in the filename_list, and then
			# access that particular portion of the file_dataset
			index = self.filename_list.index(filename)

			# any of these files belongs to Male (M) subjects - 12 in total
			if not (not (filename == "4.txt") and not (filename == "12.txt") and not (filename == "13.txt") and not (
					filename == "15.txt") and not (filename == "17.txt") and not (filename == "20.txt") and not (
					filename == "25.txt") and not (filename == "26.txt") and not (filename == "27.txt") and not (
					filename == "28.txt")) or (filename == "29.txt") or (filename == "32.txt"):
				males_dataset[i, :, :] = self.file_dataset[index]
				i = i + 1
			# the others to female files (there should be 18)
			else:
				females_dataset[j, :, :] = self.file_dataset[index]
				j = j + 1

		self.index_labeled_dataset.append(males_dataset)
		self.index_labeled_dataset.append(females_dataset)
