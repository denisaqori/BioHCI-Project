"""
Created: 3/28/19
© Denisa Qori McDonald 2019 All Rights Reserved
"""
import numpy as np
import scipy.ndimage
import math


class IntervalDescription:
	def __init__(self, interval, desc_type):
		assert desc_type == 1 or desc_type == 2
		self.interval = interval
		self.desc_type = desc_type

		self.__descriptors = None

	@property
	def descriptors(self):
		return self.produce_interval_descriptors(self.interval)

	def produce_interval_descriptors(self, interval):
		"""

		Args:
			interval: a 2D interval over which to find keypoints and produce their descriptros

		Returns:
			descriptor_list (list): list of descriptors of the interval keypoints

		"""
		descriptor_list = []
		octave = self.create_octave(interval)
		diff_of_gaussian = self.compute_diff_of_gaussian(octave)
		keypoints = self.find_keypoints(diff_of_gaussian)

		keypoint_desc = self.describe_keypoints(octave, keypoint_list=keypoints)
		if keypoint_desc is not None:
			descriptor_list.append(keypoint_desc)

		return descriptor_list

	def create_octave(self, interval):
		"""
		Blurs each column of the input interval by repeated Gaussian filtering at different scales.

		Args:
			interval: the signal whose columns are to be individually filtered

		Returns:
			octave (list): a list of intervals (ndarrays) filtered at different scales (2D)- each separately
				filtered

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

	def compute_diff_of_gaussian(self, octave):
		"""
		Computes differences of consecutive blurred signals passed in the octave list

		Args:
			octave (list of ndarray): list of progressively blurred intervals (each column individually)

		Returns:
			diff_of_gaussian_list (list of ndarray): a list of differences between consecutive input intervals

		"""

		diff_of_gaussian_list = []
		for i in range(1, len(octave)):
			diff_of_gaussian = np.subtract(octave[i], octave[i - 1])
			diff_of_gaussian_list.append(diff_of_gaussian)
		return diff_of_gaussian_list

	def find_keypoints(self, octave_dog):
		"""
		Finds points in the interval that are either the maximum or minimum of their neighbourhood within at least
		one dimension.

		Args:
			octave_dog: a list of difference of gaussian intervals

		Returns:
			signal_keypoints (list): a list of tuples - each tuple contains coordinates for a discontinuity.
									 The first coordinate indicates the scale, while the second, the time point.
		"""
		num_signal = octave_dog[0].shape[-1]

		signal_keypoints = []
		for i in range(0, num_signal):
			signal_list = []
			for interval in octave_dog:
				signal = interval[:, i]
				signal_list.append(signal)
			keypoints_1d = self._find_keypoints_1d(signal_list)
			signal_keypoints.append(keypoints_1d)

		all_keypoints = [coords for sublist in signal_keypoints for coords in sublist]
		return all_keypoints

	# admit keypoint if DoG point is smaller or larger than all the points in its neighbourhood
	def _find_keypoints_1d(self, signals_1d):
		"""

		Args:
			signals_1d:

		Returns:

		"""
		signals_1d_list = []
		for ndarray in signals_1d:
			signals_1d_list.append(ndarray.tolist())

		keypoint_idx = []
		for i in range(1, len(signals_1d_list) - 1):
			prev_signal = signals_1d_list[i - 1]
			signal = signals_1d_list[i]
			next_signal = signals_1d_list[i + 1]

			for j in range(1, len(signal) - 1):
				if (signal[j] > prev_signal[j] and signal[j] > next_signal[j] and
					signal[j] > signal[j - 1] and signal[j] > signal[j + 1] and
					signal[j] > prev_signal[j - 1] and signal[j] > prev_signal[j + 1] and
					signal[j] > next_signal[j - 1] and signal[j] > next_signal[j + 1]) or \
						(signal[j] < prev_signal[j] and signal[j] < next_signal[j] and
						 signal[j] < signal[j - 1] and signal[j] < signal[j + 1] and
						 signal[j] < prev_signal[j - 1] and signal[j] < prev_signal[j + 1] and
						 signal[j] < next_signal[j - 1] and signal[j] < next_signal[j + 1]):
					# print("Found keypoint in signal ", i, "at location ", j, "!")
					keypoint_idx.append((i, j))
		return keypoint_idx


	def describe_keypoints(self, octave, keypoint_list=None):
		"""
		Describes every point of an interval by default, unless there is a keypoint_list passed as an argument,
		in which case it uses that to find the indices of points to describe. 

		Args:
			octave (list): a list of arrays representing the same signal progressively blurred.
			keypoint_list (list): a list of tuples indicating extrama in the signal scale-space. The first element
								  of the tuple is the level of blur (scale), - indicating which of the octaves to
								  select from the list - while the second element of the tuple stands for the time
								  position within that octave that is a maximum or minimum.

		Returns:
			processed_octave (ndarray): a ndarray of 2*nb*number_of_blurred_images feature vector that describes
				each point of the interval

		"""
		descriptors = None
		nb = 4
		a = 4

		if keypoint_list is None:
			descriptors = self._compute_dense_descriptors(octave, nb, a)

		else:
			if len(keypoint_list) > 0:
				if self.desc_type == 1:
					descriptors = self._describe_keypoints_1D(octave, keypoint_list, nb, a)
				elif self.desc_type == 2:
					descriptors = self._describe_keypoints_2D(octave, keypoint_list, nb, a)

		return descriptors

	def _describe_keypoints_2D(self, octave, keypoint_list, nb, a):
		keypoint_descriptors_2D = []
		for i, keypoint in enumerate(keypoint_list):

			scale = keypoint[0]
			interval = octave[scale]
			point_idx = keypoint[1]

			neighborhood_2D_list = []
			for j in range(0, interval.shape[-1]):
				signal = interval[:, j]
				keypoint_neighbourhood = self._get_neighbourhood(signal, point_idx, nb, a)
				neighborhood_2D_list.append(keypoint_neighbourhood)

			neighborhood_2D = np.stack(neighborhood_2D_list, axis=1)
			descriptors_2D = self._describe_each_point_2D(neighborhood_2D, nb, a)
			keypoint_descriptors_2D.append(descriptors_2D)

		all_descriptors_2D = np.stack(keypoint_descriptors_2D, axis=0)
		return all_descriptors_2D

	def _describe_keypoints_1D(self, octave, keypoint_list, nb, a):
		keypoint_descriptors = []
		for i, keypoint in enumerate(keypoint_list):

			scale = keypoint[0]
			interval = octave[scale]

			point_idx = keypoint[1]
			column_desc = []
			for j in range(0, interval.shape[-1]):
				signal = interval[:, j]
				keypoint_neighbourhood = self._get_neighbourhood(signal, point_idx, nb, a)

				descriptor = self._describe_each_point_1D(keypoint_neighbourhood, nb, a)
				column_desc.append(descriptor)
			all_column_desc = np.concatenate(column_desc, axis=0)
			keypoint_descriptors.append(all_column_desc)

		all_descriptors_1D = np.stack(keypoint_descriptors, axis=0)
		return all_descriptors_1D

	def _compute_dense_descriptors(self, octave, nb, a):
		"""
		Compute descriptors for every point in each octave, instead of describing only key-points (points of change)

		Args:
			return result
			nb: number of blocks around each point to compute gradients
			a: number of intances per block

		Returns:
			descriptors(ndarray):

		"""
		processed_signals = []
		for filtered_signal in octave:
			signal_desc_list = []
			for i in range(0, octave[0].shape[-1]):
				desc = self._describe_signal_1d(filtered_signal[:, i], nb, a)
				signal_desc_list.append(desc)
			signal_desc = np.concatenate(signal_desc_list, axis=1)
			processed_signals.append(signal_desc)

		descriptors = np.concatenate(processed_signals, axis=1)
		return descriptors

	def _get_neighbourhood(self, signal_1d, point_idx, nb, a):
		"""
		Constructs a neighbourhood around a point to be described, with nb*a/2 points before and after that point.

		Args:
			signal_1d (ndarray): one column of an interval
			point_idx (int): the index of the point in the interval to be described (whose neighbourhood to get)
			nb: number of blocks to describe the point
			a: number of points in each block

		Returns:
			keypoint_neighbourhood (ndarray): a 1D array that contains the neighbourhood with the point to be
											  described in center. The size of the array is nb * a + 1.

		"""
		assert len(signal_1d) > 0

		start = int(point_idx - (nb * a) / 2)
		stop = int(point_idx + (nb * a) / 2 + 1)

		# if there aren't enough values to form blocks ahead of the keypoint, repeat the first value
		if point_idx < nb * a / 2:
			pad_n = nb * a / 2 - point_idx
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

		return keypoint_neighbourhood

	def _describe_signal_1d(self, signal_1d, nb, a):
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
		assert nb % 2 == 0, "The number of blocks that describe the keypoint needs to be even, so we can get an " \
							"equal number of points before and after the keypoint."

		keypoint_descriptors = []
		for pos, point in enumerate(signal_1d):
			keypoint_neighbourhood = self._get_neighbourhood(signal_1d, pos, nb, a)
			descriptor = self._describe_each_point_1D(keypoint_neighbourhood, nb, a)

			keypoint_descriptors.append(descriptor)

		signal_desc = np.stack(keypoint_descriptors, axis=0)
		return signal_desc

	def _describe_each_point_2D(self, keypoint_neighborhood_2D, nb, a):
		assert keypoint_neighborhood_2D is not None, "No neighbourhood has been assigned to the keypoint."
		assert keypoint_neighborhood_2D.shape[0] == nb * a + 1

		# 2 2-D arrays are returned: first standing for gradients in rows, and second for gradients in columns
		gradient_rows, gradient_cols = np.gradient(keypoint_neighborhood_2D)
		filtered_gradient_rows = scipy.ndimage.gaussian_filter(input=gradient_rows, sigma=nb * a / 2)
		filtered_gradient_cols = scipy.ndimage.gaussian_filter(input=gradient_cols, sigma=nb * a / 2)

		point_idx = int(nb * a / 2)

		all_gradients = []
		for i in range(0, filtered_gradient_rows.shape[1]):
			filtered_gradient_rows_1D = filtered_gradient_rows[:, i]
			blocks = self._create_blocks(filtered_gradient_rows_1D, point_idx, a)
			gradients_rows_1D = self._get_block_gradients(blocks)
			all_gradients.append(gradients_rows_1D)

		for i in range(0, filtered_gradient_cols.shape[1]):
			filtered_gradient_cols_1D = filtered_gradient_cols[:, i]
			blocks = self._create_blocks(filtered_gradient_cols_1D, point_idx, a)
			gradients_cols_1D = self._get_block_gradients(blocks)
			all_gradients.append(gradients_cols_1D)

		all_gradient_sums = np.concatenate(all_gradients)
		return all_gradient_sums

	"""
	def _create_2D_blocks(self, filtered_gradient_2D, point_idx, a):
		blocks = []
		for i in range(0, point_idx, a):
			block = filtered_gradient_2D[i:i + a, :]

			blocks.append(block)

		for i in range(point_idx + 1, filtered_gradient_2D.shape[0], a):
			block = filtered_gradient_2D[i:i + a, :]
			blocks.append(block)

		return blocks
	"""

	def _describe_each_point_1D(self, keypoint_neighbourhood, nb, a):
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

		blocks = self._create_blocks(filtered_gradient, point_idx, a)
		gradient_sums = self._get_block_gradients(blocks)

		return gradient_sums

	def _create_blocks(self, filtered_gradient_1D, point_idx, a):
		blocks = []
		for i in range(0, point_idx, a):
			block = filtered_gradient_1D[i:i + a]

			blocks.append(block)

		for i in range(point_idx + 1, filtered_gradient_1D.shape[0], a):
			block = filtered_gradient_1D[i:i + a]
			blocks.append(block)

		return blocks

	def _get_block_gradients(self, blocks):
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

		gradient_sums = np.array(all_gradients).flatten()
		return gradient_sums


