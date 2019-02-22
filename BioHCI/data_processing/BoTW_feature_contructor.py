"""
Created: 2/19/19
© Denisa Qori McDonald 2019 All Rights Reserved
"""
import math

from BioHCI.data_processing.feature_constructor import FeatureConstructor
from BioHCI.helpers.study_config import StudyConfig
from BioHCI.data.data_constructor import DataConstructor
from BioHCI.data_processing.dataset_processor import DatasetProcessor
import scipy.ndimage
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
			filtered = scipy.ndimage.gaussian_filter1d(input=signal, sigma=sigma)
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

	# TODO: concatenate results from all octaves
	def _describe_keypoints(self, octave):
		"""
		Describes every point of an interval.

		Args:
			octave:

		Returns:
			processed_octave (ndarray): a ndarray of 2*nb*number_of_blurred_images feature vector that describes
				each point of the interval

		"""
		processed_signals = []
		for filtered_signal in octave:
			signal_desc_list = []
			for i in range(0, octave[0].shape[-1]):
				desc = self._describe_signal_1d(filtered_signal[:, i])
				signal_desc_list.append(desc)
			signal_desc = np.concatenate(signal_desc_list, axis=1)
			processed_signals.append(signal_desc)

		processed_octave = np.concatenate(processed_signals, axis=1)
		return

	def _describe_signal_1d(self, signal_1d, nb=4, a=4):
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

			start = int(pos - (nb * a) / 2)
			stop = int(pos + (nb * a) / 2 + 1)

			# if there aren't enough values to form blocks ahead of the keypoint, repeat the first value
			if pos < nb * a / 2:
				pad_n = nb * a / 2 - pos
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
				keypoint_neighbourhood = signal_1d[start: stop]

			descriptor = self._describe_each_point(keypoint_neighbourhood, nb=nb, a=a)
			keypoint_descriptors.append(descriptor)

		signal_desc = np.stack(keypoint_descriptors, axis=0)
		return signal_desc


	# TODO: make sure that there is no bug where all negative gradient values are smoothed out
	def _describe_each_point(self, keypoint_neighbourhood, nb, a):
		"""
		Each keypoint is described in terms of the sum of positive and negative gradients of blocks of other points
		around it. The interval keypoint_neighbourhood, whose midpoint is the keypoint being characterized,
		is first filtered with a gaussian filter of scale nb*a/2 to weigh the importance of points according to
		proximity with the keypoint.

		Args:
			keypoint_neighbourhood:
			nb: the total number of blocks around a keypoint
			a: the number of points in one block

		Returns:
			all_gradient_sums (ndarray): a 2*nb long ndarray where from each block, the sum of positive gradients,
			and the sum of negative gradients are maintained

		"""
		assert keypoint_neighbourhood is not None, "No neighbourhood has been assigned to the keypoint."
		assert keypoint_neighbourhood.shape[0] == nb * a + 1

		gradient = np.gradient(keypoint_neighbourhood)
		filtered_gradient = scipy.ndimage.gaussian_filter1d(input=gradient, sigma=nb * a / 2)
		point_idx = int(nb * a / 2)

		blocks = []
		for i in range(0, point_idx, a):
			block = filtered_gradient[i:i+a]

			blocks.append(block)

		for i in range(point_idx+1, filtered_gradient.shape[0], a):
			block = filtered_gradient[i:i+a]
			blocks.append(block)

		all_gradients = []
		for j, block in enumerate(blocks):
			pos = 0
			neg = 0
			for point in block:
				if point < 0:
					neg = neg + point
				if point > 0:
					pos = pos + point
			all_gradients.append([pos, neg])

		all_gradient_sums = np.array(all_gradients).flatten()
		return all_gradient_sums


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
