"""
Created: 3/27/19
© Denisa Qori McDonald 2019 All Rights Reserved
"""
from BioHCI.data_processing.BoTW_feature_contructor import BoTWFeatureConstructor
import pickle

class DescriptorEvaluator:
	def __init__(self, dataset_desc_path):
		self.__dataset_descriptors = self.load_descriptors(dataset_desc_path)

	@property
	def dataset_descriptors(self):
		assert self.dataset_descriptors is not None
		return self.__dataset_descriptors

	def load_descriptors(self, dataset_desc_path):
		with open(dataset_desc_path, "rb") as input_file:
			dataset_desc = pickle.load(input_file)
		return dataset_desc

	def compute_heatmap(self):
		return

	def levenshtein_distance(self, keypress1, keypress2):
		return