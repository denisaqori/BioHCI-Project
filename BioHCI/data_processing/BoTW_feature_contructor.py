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
		"""

		Args:
			cat:
			feature_axis:

		Returns:

		"""
		for i in range(0, cat.shape[0]):
			for j in range(0, cat.shape[1]):
				interval = cat[i, j, :, :]
				octave = self._create_octave(interval)
				diff_of_gaussian = self._compute_diff_of_gaussian(octave)
				self._describe_keypoints(octave)

	def _create_octave(self, interval):
		"""
		Blurs each column of the input interval by repeated Gaussian filtering at different scales.

		Args:
			interval: the signal whose columns are to be individually filtered

		Returns:
			octave (list): a list of intervals (ndarrays) filtered at different scales (2D)- each separately filtered

		"""
		k = math.sqrt(2)
		sigma = 0.5

		octave = []
		for i in range(0, 5):
			sigma = k * sigma
			smoothed_interval = self._smooth_interval(interval, sigma)
			octave.append(smoothed_interval)

		return octave

	def _smooth_interval(self, interval, sigma):
		"""
		Each column of the input interval is smoothed at a particular scale.

		Args:
			interval: the signal whose columns are to be individually filtered
			sigma: smoothing scale

		Returns:
			smoothed_interval (ndarray): filtered 2D array at sigma scale

		"""
		signal_list = []
		for i in range(0, interval.shape[-1]):
			signal = interval[:, i]
			filtered = filter.gaussian_filter1d(input=signal, sigma=sigma)
			signal_list.append(filtered)

		smoothed_interval = np.stack(signal_list, axis=1)
		return smoothed_interval

	def _compute_diff_of_gaussian(self, octave):
		"""
		Computes differences of consecutive blurred signals passed in the octave list

		Args:
			octave (list of ndarray): list of progressively blurred intervals (each column individually)

		Returns:
			diff_of_gaussian_list (list of ndarray): a list of differences between consecutive input intervals

		"""

		diff_of_gaussian_list = []
		for i in range(1, len(octave)):
			diff_of_gaussian = np.subtract(octave[i - 1], octave[i])
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

	def _describe_signal(self, signal_1d, nb=4, a=4):
		"""
		Describes each point of the input signal in terms of positive and negative gradients in its neighbourhood.

		Args:
			signal_1d (ndarray): a 1-dimensional signal, each point of which will be described in terms of gradients
				of its surrounding blocks
			nb: the total number of blocks to consider per point (half on each side) where positive and negative
				gradients' sums will be computed
			a: the number of points in each block

		Returns (ndarray): a numpy array of all points where each is described by 2 values per block (2*nb feature
			vector)

		"""
		assert nb % 2 == 0, "The number of blocks that describe the keypoint needs to be even, so we can get an equal " \
							"number of points before and after the keypoint."

		keypoint_descriptors = []
		for pos, point in enumerate(signal_1d):

			start = int(pos - (nb*a)/2)
			stop = int(pos + (nb*a)/2 + 1)

			# if there aren't enough values to form blocks ahead of the keypoint, repeat the first value
			if pos < nb*a/2:
				pad_n = nb*a/2 - pos
				padding = np.repeat(signal_1d[0], pad_n)
				signal = signal_1d[0: stop]

				keypoint_neighbourhood = np.concatenate((padding, signal), axis=0)

			# if there aren't enough values to form blocks after the keypoint, repeat the last value
			elif signal_1d.shape[0] < stop:
				signal = signal_1d[start: stop]

				pad_n = stop - signal_1d.shape[0]
				padding = np.repeat(signal_1d[-1], pad_n)
				keypoint_neighbourhood = np.concatenate((signal, padding), axis=0)

			else:
				keypoint_neighbourhood = signal_1d[start : stop]

			descriptor = self._describe_each_point(keypoint_neighbourhood, nb=nb, a=a)
			keypoint_descriptors.append(descriptor)

		print("")


	def _describe_each_point(self, keypoint_neighbourhood, nb, a):
		assert keypoint_neighbourhood is not None, "No neighbourhood has been assigned to the keypoint."
		return keypoint_neighbourhood



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
