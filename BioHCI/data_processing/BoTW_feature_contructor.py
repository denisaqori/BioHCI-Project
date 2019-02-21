"""
Created: 2/19/19
© Denisa Qori McDonald 2019 All Rights Reserved
"""
import math

from BioHCI.data_processing.feature_constructor import FeatureConstructor
from BioHCI.helpers.study_config import StudyConfig
from BioHCI.data.data_constructor import DataConstructor
from BioHCI.data_processing.dataset_processor import DatasetProcessor
import scipy.ndimage as filter
import numpy as np


class BoTWFeatureConstructor(FeatureConstructor):
	"""
	Bag of Temporal Words:
	"""
	def __init__(self, parameters, feature_axis):
		super().__init__(parameters, feature_axis)
		print("Bag of Temporal Words being initiated...")
		self.features = [self.smooth_category]

	def smooth_category(self, cat, feature_axis):
		for i in range(0, cat.shape[0]):
			for j in range(0, cat.shape[1]):
				interval = cat[i, j, :, :]
				octave = self._create_octave(interval)
				diff_of_gaussian = self._compute_diff_of_gaussian(octave)
				self._describe_keypoints(octave)

	def _create_octave(self, interval):
		k = math.sqrt(2)
		sigma = 0.5

		octave = []
		for i in range (0, 5):
			sigma = k*sigma
			smoothed_interval = self._smooth_interval(interval, sigma)
			octave.append(smoothed_interval)

		return octave

	def _smooth_interval(self, interval, sigma):
		signal_list = []
		for i in range(0, interval.shape[-1]):
			signal = interval[:, i]
			filtered = filter.gaussian_filter1d(input=signal, sigma=sigma)
			signal_list.append(filtered)

		smoothed_interval = np.stack(signal_list, axis=1)
		return smoothed_interval

	def _compute_diff_of_gaussian(self, octave):

		diff_of_gaussian_list = []
		for i in range(1, len(octave)):
			diff_of_gaussian = np.subtract(octave[i-1], octave[i])
			diff_of_gaussian_list.append(diff_of_gaussian)
		return diff_of_gaussian_list

	def _describe_keypoints(self, octave):
		for filtered_signal in octave:
			gradient_list = []
			for i in range(0, octave[0].shape[-1]):
				desc = self._describe_signal(filtered_signal[:, i])
				# gradient = np.gradient(filtered_signal[:, i])
				# gradient_list.append(gradient)
			# signal_gradient = np.stack(gradient_list, axis=1)
			# print("")

	#TODO: for each point of the signal call another function that implements the 4 steps of describing each point
	def _describe_signal(self, signal_1d, nb=4, a=4):
		return

	def _describe_each_point(self):
		return

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

	feature_constructor = BoTWFeatureConstructor(parameters, feature_axis=2)
	dataset_processor = DatasetProcessor(parameters, feature_constructor=feature_constructor)
	feature_dataset = dataset_processor.process_dataset(subject_dict)
	print("")
