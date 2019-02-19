"""
Created: 2/19/19
© Denisa Qori McDonald 2019 All Rights Reserved
"""

from BioHCI.data_processing.feature_constructor import FeatureConstructor
from BioHCI.helpers.study_config import StudyConfig
from BioHCI.data.data_constructor import DataConstructor
from BioHCI.data_processing.dataset_processor import DatasetProcessor
import numpy as np

class StatFeatureConstructor(FeatureConstructor):
	"""
	Statistical Information:
	"""
	def __init__(self, parameters, feature_axis):
		super().__init__(parameters, feature_axis)
		print("Statistical Feature Constructor being initiated.")

		# methods to calculate particular features
		self.features = [self.min_features, self.max_features, self.mean_features, self.std_features]

	def min_features(self, cat, feature_axis):
		min_array = np.amin(cat, axis=feature_axis, keepdims=False)
		return min_array

	def max_features(self, cat, feature_axis):
		max_array = np.amax(cat, axis=feature_axis, keepdims=False)
		return max_array

	def mean_features(self, cat, feature_axis):
		mean_array = np.mean(cat, axis=feature_axis, keepdims=False)
		return mean_array

	def std_features(self, cat, feature_axis):
		std_array = np.std(cat, axis=feature_axis, keepdims=False)
		return std_array

	def diff_log_mean(self, cat, feature_axis):
		mean_array =self.mean_features(cat, feature_axis)
		assert mean_array.shape[-1] == 2
		diff = np.log(mean_array[:, :, 0]) - np.log(mean_array[:, :, 1])

		diff = np.expand_dims(diff, axis=feature_axis)
		return diff

if __name__ == "__main__":
	print("Running feature_constructor module...")

	config_dir = "config_files"
	config = StudyConfig(config_dir)

	# create a template of a configuration file with all the fields initialized to None
	config.create_config_file_template()
	parameters = config.populate_study_parameters("CTS_one_subj_firm.toml")

	# generating the data from files
	data = DataConstructor(parameters)
	subject_dict = data.get_subject_dataset()

	feature_constructor = StatFeatureConstructor(parameters, feature_axis=2)
	dataset_processor = DatasetProcessor(parameters, feature_constructor=feature_constructor)
	feature_dataset = dataset_processor.process_dataset(subject_dict)
	print("")
