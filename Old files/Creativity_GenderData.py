from BioHCI.data.data_constructor import AbstractData
import numpy as np

class Creativity_GenderData(AbstractData):

	def get_dataset_name(self):
		return "Gender Data"

	# information obtained from file
	def create_categories(self):
		print("\nCreativity Gender Data Processing object is being initialized...")
		self.categories.append('M')
		self.categories.append('F')
		self.num_males = 2
		self.num_females = 5

	def index_dataset_by_label(self):

		print("Creating the dataset by stacking subject data of the same label in the same numpy array.")

		# create numpy arrays for each class, to be populated by the file information below
		males_dataset = np.zeros(shape=(self.num_males, self.__inst_per_subj, self.__num_features))
		females_dataset = np.zeros(shape=(self.num_females, self.__inst_per_subj, self.__num_features))

		i = 0
		j = 0
		for filename in self.filename_list:

			# for the given filename, find the index in the filename_list, and then
			# access that particular portion of the file_dataset
			index = self.filename_list.index(filename)

			# any of these files belongs to Male (M) subjects - in total
			if (filename == "116-CB-AG-fnirs.txt") or (filename =="109-CB-JS-fnirs.txt"):
				males_dataset[i, :, :] = self.file_dataset[index]
				i = i + 1
			# the others to female files (there should be )
			elif ((filename == "104-CB-AW-fnirs.txt") or (filename == "106-CB-DM-fnirs.txt")
				  or (filename == "105-CB-JK-fnirs.txt") or (filename == "112-CB-DS-fnirs.txt")
				  or (filename =="117-CB-SK-fnirs.txt")):
				females_dataset[j, :, :] = self.file_dataset[index]
				j = j + 1

		self.index_labeled_dataset.append(males_dataset)
		self.index_labeled_dataset.append(females_dataset)