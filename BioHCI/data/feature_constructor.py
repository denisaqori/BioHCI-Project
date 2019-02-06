import numpy as np
from copy import copy
from BioHCI.helpers.study_config import StudyConfig
from BioHCI.data.data_constructor import DataConstructor
from BioHCI.data.dataset_processor import DatasetProcessor


class FeatureConstructor:
	def __init__(self, parameters, feature_axis):

		assert (parameters.construct_features is True)
		assert (parameters.feature_window is not None), "In order for features to be created, the feature window " \
														"attribute should be set to an integer greater than 0, " \
														"and be of NoneType."
		assert isinstance(feature_axis, int)
		self.feature_axis = feature_axis
		self.feature_window = parameters.feature_window
		# methods to calculate particular features
		self.features = [self.min_features, self.max_features, self.mean_features, self.std_features]

	def get_feature_dataset(self, subj_dataset):
		"""
		Constructs features over an interval of a chunk for the whole dataset. For each unprocessed original feature,
		the min, max, mean, std of the parts of each chunk are calculated.

		Args:
			subj_dataset: A dictionary mapping a subject name to a Subject object. The data for each category of each
				subject will have a shape of (number of chunks, number of feature intervals, interval to built
				features on, number of original attributes).
			feature_axis: The axis along which features are to be created. This axis will end up being collapsed

		Returns:
			feature_dataset: A dictionary mapping a subject name to a Subject object. The data for each category of
				this subject object will have been processed to have some features calculated, and will have the
				shape: (number of chunks, instances per chunk, features (previous attributes * number of features
				calculated for each).

		"""
		assert isinstance(subj_dataset, dict)

		feature_dataset = {}
		for subj_name, subj in subj_dataset.items():
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

		return feature_dataset

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

	# TODO: see if this serves a purpose at some point - right now, not any
	# this function changes the view of a dataset which is chunked by time (by merging the first two dimensions)
	# as well as the corresponding labels. Useful in case one measurement is input as an observation without
	# constructing
	# features over a specific time window. Input is a tensor, output a numpy array
	# especially helpful for a traditional ML algorithm not needing constructed features
	def unify_time_windows(time_chunked_dataset, time_chunked_labels, samples_per_step):
		print("Unifying time windows...")
		print("time_chunked_dataset: ", time_chunked_dataset)
		print("time_chunked_labels: ", time_chunked_labels)

		labels_tuple = ()
		for label in time_chunked_labels:
			seq_labels = int(label) * np.ones(samples_per_step)
			labels_tuple = labels_tuple + (seq_labels,)
		labels = np.cat(labels_tuple)

		dataset = time_chunked_dataset.view(time_chunked_dataset.size(0) * time_chunked_dataset.size(1),
											time_chunked_dataset.size(2)).numpy()

		print("unifiedDataset shape: ", dataset.shape)
		print("unifiedLabels shape: ", labels.shape)

		return dataset, labels


if __name__ == "__main__":
	print("Running feature_constructor module...")

	config_dir = "config_files"
	config = StudyConfig(config_dir)

	# create a template of a configuration file with all the fields initialized to None
	config.create_config_file_template()
	parameters = config.populate_study_parameters("CTS_one_subj_variable.toml")

	# generating the data from files
	data = DataConstructor(parameters)
	subject_dict = data.get_subject_dataset()

	feature_constructor = FeatureConstructor(parameters)
	dataset_processor = DatasetProcessor(parameters, feature_constructor=feature_constructor)
	feature_dataset = dataset_processor.process_dataset(subject_dict)
