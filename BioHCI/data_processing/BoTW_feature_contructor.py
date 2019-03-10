"""
Created: 2/19/19
© Denisa Qori McDonald 2019 All Rights Reserved
"""
import math

from BioHCI.data_processing.feature_constructor import FeatureConstructor
from BioHCI.helpers.study_config import StudyConfig
from BioHCI.data.data_constructor import DataConstructor
from BioHCI.data_processing.dataset_processor import DatasetProcessor
import BioHCI.helpers.utilities as utils
from BioHCI.data_processing.within_subject_oversampler import WithinSubjectOversampler

import torch
import time
from multiprocessing.dummy import Pool as ThreadPool
from multiprocessing import Pool
from sklearn.cluster import KMeans
import scipy.ndimage
import numpy as np
import pickle
import os


# TODO: run multi-threaded: disabled for the moment so there is some reproducibility in results
class BoTWFeatureConstructor(FeatureConstructor):
	"""
	Bag of Temporal Words:
	"""

	def __init__(self, dataset_processor, parameters, feature_axis, codebook_name=None):
		super().__init__(dataset_processor, parameters, feature_axis)
		print("Bag of Temporal Words being initiated...")

		self.codebook_name = None
		codebooks_path = '/home/denisa/GitHub/BioHCI Project/BioHCI/data_processing/codebooks'
		self.all_codebooks_dir = utils.create_dir(codebooks_path)
		self.features = [self.compute_histogram]

	# TODO: make the way the axis is extracted more general
	def compute_histogram(self, cat, feature_axis):
		"""
		For each interval of each chunk, the number of keypoints that belong to each cluster is calculated,
		and the distribution over all the clusters for each interval is converted to a 1D feature vector.

		Args:
			cat: the category over which to create features in the form form of distributions over clusters

		Returns:
			category_dist (ndarray): a 3D ndarray containing distribution of keypoints over different clusters. The
			shape is (nubmer of chunks, instances per chunk, number of clusters(=number of final attributes)).

		"""
		# load the model
		assert self.codebook_name is not None
		codebook_path = os.path.abspath(os.path.join(self.all_codebooks_dir, self.codebook_name))
		assert os.path.exists(codebook_path)

		kmeans = pickle.load(open(codebook_path, 'rb'))

		new_chunk_list = []
		for i in range(0, cat.shape[0]):
			chunk_distributions = []
			for j in range(0, cat.shape[1]):
				interval = cat[i, j, :, :]
				interval_descriptors_list = self._produce_interval_descriptors(interval)

				if len(interval_descriptors_list) > 0:
					interval_descriptors = np.concatenate(interval_descriptors_list, axis=0)
					prediction = kmeans.predict(interval_descriptors)
					interval_distribution = self._compute_interval_distribution(prediction, kmeans.n_clusters)
				else:
					interval_distribution = np.zeros((1, kmeans.n_clusters))

				chunk_distributions.append(interval_distribution)
			new_chunk = np.concatenate(chunk_distributions, axis=0)
			new_chunk_list.append(new_chunk)
		category_dist = np.stack(new_chunk_list, axis=0)
		return category_dist

	def _compute_interval_distribution(self, prediction, num_clusters):
		"""

		Args:
			prediction (ndarray): a ndarray whose elements contain the indices of the cluster centers to which a
								  descriptor belongs
			num_clusters (int): the total number of clusters in the KMeans algorithm

		Returns:
			dist(ndarray): a ndarray of dimension (1, num_clusters) where each index represents the particular
							cluster and its value, the number of descriptors in that interval that belong to that
							cluster.
		"""
		dist = np.zeros((1, num_clusters))
		for i in range(0, prediction.shape[0]):
			for x in np.nditer(prediction[i]):
				dist[0, x] = dist[0, x] + 1

		return dist

	def generate_codebook(self, subj_dataset, codebook_name):
		"""

		Args:
			subj_dataset:

		Returns:

		"""
		codebook_path = os.path.abspath(os.path.join(self.all_codebooks_dir, codebook_name))
		if not os.path.exists(codebook_path):

			cat_desc_list = []
			for subj_name, subj in subj_dataset.items():
				cat_data = subj.get_data()
				for cat in cat_data:
					cat_desc = self.produce_category_descriptors(cat)

					# multi-threading (easily done by mapping)
					# pool = Pool(10)
					# cat_desc = pool.map(self.produce_category_descriptors, cat_data)
					# pool.close()
					# pool.join()

					if cat_desc is not None:
						cat_desc_list.append(cat_desc)
			dataset_desc = np.concatenate(cat_desc_list, axis=0)
			kmeans = KMeans(n_clusters=10).fit(dataset_desc)

			# save the model to disk
			pickle.dump(kmeans, open(codebook_path, 'wb'))
		else:
			print ("Codebook: ", codebook_name, "already exists in ", self.all_codebooks_dir)
		self.codebook_name = codebook_name

	def produce_category_descriptors(self, cat):
		"""

		Args:
			cat (ndarray): category to be described

		Returns:
			cat_descriptors:

		"""
		# cat = torch.from_numpy(cat).cuda(async=True)
		descriptor_list = []
		for i in range(0, cat.shape[0]):
			for j in range(0, cat.shape[1]):
				interval = cat[i, j, :, :]
				interval_desc_list = self._produce_interval_descriptors(interval)
				descriptor_list.append(interval_desc_list)

		descriptor_list = [desc for sublist in descriptor_list for desc in sublist]
		cat_descriptors = np.concatenate(descriptor_list, axis=0)
		return cat_descriptors

	def _produce_interval_descriptors(self, interval):
		"""

		Args:
			interval: a 2D interval over which to find keypoints and produce their descriptros

		Returns:
			descriptor_list (list): list of descriptors of the interval keypoints

		"""
		descriptor_list = []
		octave = self._create_octave(interval)
		diff_of_gaussian = self._compute_diff_of_gaussian(octave)
		keypoints = self._find_keypoints(diff_of_gaussian)

		keypoint_desc = self._describe_keypoints(octave, keypoint_list=keypoints)
		if keypoint_desc is not None:
			descriptor_list.append(keypoint_desc)

		return descriptor_list

	def _create_octave(self, interval):
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
			diff_of_gaussian = np.subtract(octave[i], octave[i - 1])
			diff_of_gaussian_list.append(diff_of_gaussian)
		return diff_of_gaussian_list

	def _find_keypoints(self, octave_dog):
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
					print("Found keypoint in signal ", i, "at location ", j, "!")
					keypoint_idx.append((i, j))
		return keypoint_idx

	def _describe_keypoints(self, octave, keypoint_list=None):
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
		nb = 4
		a = 4

		if keypoint_list is None:
			processed_signals = []
			for filtered_signal in octave:
				signal_desc_list = []
				for i in range(0, octave[0].shape[-1]):
					desc = self._describe_signal_1d(filtered_signal[:, i], nb, a)
					signal_desc_list.append(desc)
				signal_desc = np.concatenate(signal_desc_list, axis=1)
				processed_signals.append(signal_desc)

			result = np.concatenate(processed_signals, axis=1)

		else:
			if len(keypoint_list) > 0:
				keypoint_descriptors = []
				for i, keypoint in enumerate(keypoint_list):

					print("keypoint: ", keypoint)
					scale = keypoint[0]
					interval = octave[scale]

					point_idx = keypoint[1]
					column_desc = []
					for j in range(0, interval.shape[-1]):
						signal = interval[:, j]
						keypoint_neighbourhood = self._get_neighbourhood(signal, point_idx, nb, a)
						descriptor = self._describe_each_point(keypoint_neighbourhood, nb, a)
						column_desc.append(descriptor)
					all_column_desc = np.concatenate(column_desc, axis=0)
					keypoint_descriptors.append(all_column_desc)

				# keypoint_desc = np.stack(keypoint_descriptors, axis=0)
				# keypoint_desc_list.append(keypoint_desc)
				result = np.stack(keypoint_descriptors, axis=0)
			else:
				result = None

		return result

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
			descriptor = self._describe_each_point(keypoint_neighbourhood, nb, a)
			keypoint_descriptors.append(descriptor)

		signal_desc = np.stack(keypoint_descriptors, axis=0)
		return signal_desc

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
			block = filtered_gradient[i:i + a]

			blocks.append(block)

		for i in range(point_idx + 1, filtered_gradient.shape[0], a):
			block = filtered_gradient[i:i + a]
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
	print("Is cuda available?", torch.cuda.is_available())

	config_dir = "config_files"
	config = StudyConfig(config_dir)

	# create a template of a configuration file with all the fields initialized to None
	config.create_config_file_template()
	parameters = config.populate_study_parameters("CTS_one_subj_firm.toml")

	# generating the data from files
	data = DataConstructor(parameters)
	subject_dict = data.get_subject_dataset()

	# define a category balancer (implementing the abstract CategoryBalancer)
	category_balancer = WithinSubjectOversampler()
	dataset_processor = DatasetProcessor(parameters, balancer=category_balancer)

	feature_constructor = BoTWFeatureConstructor(dataset_processor, parameters, feature_axis=2)
	feature_constructor.generate_codebook(subject_dict, "bag_of_temporal_words_codebook.sav")
	feature_dataset = feature_constructor.produce_feature_dataset(subject_dict)
	print("Done")
