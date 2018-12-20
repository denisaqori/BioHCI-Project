import numpy as np
from copy import copy


# TODO: look into not passing all the parameters object - maybe only the feature window? Maybe not.
class FeatureConstructor:
	def __init__(self, parameters):

		print("Feature construction not implemented yet.... Should be explicitly called after done with data "
			  "splitting, slicing, balancing.")

		assert (parameters.construct_features is True)
		assert (parameters.feature_window is not None), "In order for features to be created, the feature window " \
														"attribute should be set to an integer greater than 0, and not " \
														"be of NoneType."
		self.feature_window = parameters.feature_window

	# TODO: deal with all_data_bool within Subject (for this and other functions that copy and create a new subj)
	def get_stat_features(self, subj_dataset, feature_axis):
		"""
		Constructs features over an interval of a chunk for the whole dataset. For each unprocessed original feature,
		the min, max, mean, std of the parts of each chunk are calculated.

		Args:
			subj_dataset: A dictionary mapping a subject name to a Subject object. The data for each category of each
				subject will have a shape of (number of chunks, number of feature intervals, interval to built
				features on, number of original attributes).
			feature_axis: The axis along which features are to be created. This axis will end up being collapsed

		Returns:

		"""
		assert isinstance(subj_dataset, dict)
		assert isinstance(feature_axis, int)

		feature_dataset = {}
		for subj_name, subj in subj_dataset.items():
			cat_data = subj.get_data()
			cat_names = subj.get_categories()

			new_cat_data = []
			for cat in cat_data:
				assert len(cat.shape) == 4, "The subj_dataset passed to create features on, should have 4 axis."
				assert feature_axis in range(0, len(cat.shape))
				min_array = np.amin(cat, axis=feature_axis, keepdims=True)
				max_array = np.amax(cat, axis=feature_axis, keepdims=True)
				mean_array = np.mean(cat, axis=feature_axis, keepdims=True)
				std_array = np.std(cat, axis=feature_axis, keepdims=True)

				new_cat = np.concatenate((min_array, max_array, mean_array, std_array), axis=feature_axis)
				new_cat = np.reshape(new_cat, newshape=(new_cat.shape[0], new_cat.shape[1], -1))
				new_cat_data.append(new_cat)

			new_subj = copy(subj)  # copy the current subject
			new_subj.set_data(new_cat_data)  # assign the above-calculated feature categories
			feature_dataset[subj_name] = new_subj  # assign the Subject object to its name (unaltered)

		return feature_dataset

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
			seq_labels = int(label) * torch.ones(samples_per_step)
			labels_tuple = labels_tuple + (seq_labels,)
		labels = torch.cat(labels_tuple).numpy()

		dataset = time_chunked_dataset.view(time_chunked_dataset.size(0) * time_chunked_dataset.size(1),
											time_chunked_dataset.size(2)).numpy()

		print("unifiedDataset shape: ", dataset.shape)
		print("unifiedLabels shape: ", labels.shape)

		return dataset, labels
