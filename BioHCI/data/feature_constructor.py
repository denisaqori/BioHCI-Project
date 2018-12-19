import numpy as np

# TODO: implement FeatureConstructor Class
# TODO: look into not passing all the parameters object - maybe only the feature window? Maybe not.
class FeatureConstructor:
	def __init__(self, parameters):

		print("Feature construction not implemented yet.... Should be explicitly called after done with data "
			  "splitting, slicing, balancing.")

		assert(parameters.construct_features is True), ""
		assert(parameters.feature_window is not None), "In order for features to be created, the feature window attribute " \
												 "should be set to an integer greater than 0, and not be of NoneType."
		self.feature_window = parameters.feature_window

	def construct_features(self, subj_dataset):
	# tp

		for subj_name, subj in subj_dataset.items():
			cat_data = subj.get_data()
			cat_names = subj.get_categories()

			for cat in cat_data:
				min_array = np.amin(cat, axis=2)

		return self.feature_window


	# this function constructs features over a chunk, which contains samples_per_step measurements
	# for that time window, for each unprocessed original feature, the min, max, mean, std of the
	# samples_per_step is calculated
	def define_standard_features(dataset):
		print("Constructing features....")

		# for testing purposes - making sure that the values of mean, max, min, and std are properly calculated over
		# dimension 1 (which should have samples_per_step values)
		print("Dataset[0,:,:]", dataset[0, :, :])

		# setting keepdim to false for each of these calculations, removes the dimension along which they were calculated
		# for min and max we keep only the first value of the tuple returned, which contains the actual min, and max values
		# respectively, while the second one gives the index of those values in the dimension along which they were calculated
		min_tensor = dataset.min(dim=1, keepdim=False)[0]
		print("Min tensor: ", min_tensor)

		max_tensor = dataset.max(dim=1, keepdim=False)[0]
		print("Max tensor: ", max_tensor)

		mean_tensor = dataset.mean(dim=1, keepdim=False)
		print("Mean tensor: ", mean_tensor)

		std_tensor = dataset.std(dim=1, keepdim=False)
		print("Standard deviation tensor: ", std_tensor)

		feature_dataset = torch.cat((min_tensor, max_tensor, mean_tensor, std_tensor), dim=1)
		print("New dataset with constructed features: ", feature_dataset)

		return feature_dataset, labels


	#TODO: see if this serves a purpose at some point - right now, not any
	# this function changes the view of a dataset which is chunked by time (by merging the first two dimensions)
	# as well as the corresponding labels. Useful in case one measurement is input as an observation without constructing
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

