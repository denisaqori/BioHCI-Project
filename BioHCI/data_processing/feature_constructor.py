import numpy as np
from copy import copy
from abc import ABC
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import BioHCI.helpers.utilities as utils

class FeatureConstructor(ABC):
	def __init__(self, dataset_processor, parameters, feature_axis):

		assert (parameters.construct_features is True)
		assert (parameters.feature_window is not None), "In order for features to be created, the feature window " \
														"attribute should be set to an integer greater than 0, " \
														"and be of NoneType."
		self.parameters = parameters
		assert isinstance(feature_axis, int)
		# A dictionary mapping a subject name to a Subject object. The data for each category of each subject will
		# have a shape of (number of chunks, number of feature intervals, interval to built features on, number of
		# original attributes).

		# The axis along which features are to be created. This axis will end up being collapsed
		self.feature_axis = feature_axis
		self.feature_window = parameters.feature_window
		self.dataset_processor = dataset_processor

	def _process_dataset(self, subject_dataset):
		"""
		Runs dataset_processor on the subject dataset, and additionally chunks it to create an axis over which
		features will be built, and which will be eventually collapsed.

		Returns:

		"""
		# if (self.parameters.chunk_instances is True):
		assert subject_dataset is not None, "subject_dataset needs to be set."
		processed_dataset = self.dataset_processor.process_dataset(subject_dataset)

		feature_ready_dataset = self.dataset_processor.chunk_data(processed_dataset, self.parameters.feature_window,
																  1, self.parameters.feature_overlap)
		return feature_ready_dataset
		# else:
		# 	return subject_dataset

	@property
	def features(self):
		return self.__features

	@features.setter
	def features(self, feature_list):
		assert isinstance(feature_list, list)
		for feature in feature_list:
			assert callable(feature), "The variable feature_list needs to contain functions that compute features."

		self.__features = feature_list

	def produce_feature_dataset(self, subject_dataset):
		"""
		Constructs features over an interval of a chunk for the whole dataset. For each unprocessed original feature,
		the function specified in self.features is applied to each part of the chunk. First the existence of
		self.subject_dataset is checked to ensure it has been assigned. At the end of the processing,
		self.subject_dataset is reset to None, so the same FeatureConstructor definition can be used on another
		subject dataset.

		Returns:
			feature_dataset: A dictionary mapping a subject name to a Subject object. The data for each category of
				this subject object will have been processed to have some features calculated, and will have the
				shape: (number of chunks, instances per chunk, features (previous attributes * number of features
				calculated for each).

		"""
		assert subject_dataset is not None, "A subject dataset needs to be set to this FeatureConstructor object."
		assert isinstance(subject_dataset, dict)
		assert self.features is not None, "There features should contain functions that calculate features in a " \
										  "specific way. Initiate a non-abstract class, child of FeatureConstructor."

		processed_dataset = self._process_dataset(subject_dataset)
		feature_dataset = {}
		for subj_name, subj in processed_dataset.items():
			cat_data = subj.get_data()

			new_cat_data = []
			for cat in cat_data:
				assert len(cat.shape) == 4, "The subj_dataset passed to create features on, should have 4 axis."
				assert self.feature_axis in range(-1, len(cat.shape))

				feature_cat_tuple = ()
				for feature_func in self.features:
					features = feature_func(cat, self.feature_axis)

					feature_cat_tuple = feature_cat_tuple + (features,)

				new_cat = np.stack(feature_cat_tuple, axis=self.feature_axis)

				new_cat_reshaped = np.reshape(new_cat, newshape=(new_cat.shape[0], new_cat.shape[1], -1))
				new_cat_data.append(new_cat_reshaped)

			new_subj = copy(subj)  # copy the current subject
			new_subj.set_data(new_cat_data)  # assign the above-calculated feature categories
			feature_dataset[subj_name] = new_subj  # assign the Subject object to its name (unaltered)

		# reset self.subject_dataset to None, so another subject dataset can be assigned to the feature definition
		self.subject_dataset = None
		return feature_dataset

	'''
	def sift(self, cat, feature_axis):
		sift = cv2.xfeatures2d.SIFT_create(nfeatures=3, contrastThreshold = 0.001, edgeThreshold = 200)

		for i in range(0, cat.shape[0]):
			for j in range(0, cat.shape[1]):
				interval = cat[i, j, :, :].astype(np.uint8)
				kp = sift.detect(interval, None)
				print("")
		print("")

	def surf(self, cat, feature_axis):
		surf = cv2.xfeatures2d.SURF_create()

		for i in range(0, cat.shape[0]):
			for j in range(0, cat.shape[1]):
				interval = cat[i, j, :, :].astype(np.uint8)
				kp = surf.detect(interval, None)
				print("")

	def hog(self, cat, feature_axis):
		hog = cv2.HOGDescriptor(blockSize=(10,1), blockStride=(5,1), cellSize=(5,1))

		for i in range(0, cat.shape[0]):
			for j in range(0, cat.shape[1]):
				interval = cat[i, j, :, :]
				descriptor = hog.compute(interval)
				if descriptor is None:
					descriptor = []
				else:
					descriptor = descriptor.ravel()
				print("")
	'''
