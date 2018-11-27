import math
import time
import datetime

import torch
import os
import errno


# this function calculates timing difference to measure how long running certain parts takes
def time_since(since):
	now = time.time()
	s = now - since
	m = math.floor(s / 60)
	s -= m * 60
	return '%dm %ds' % (m, s)

def time_now():
	#return datetime.now().strftime("%Y%m%d-%H%M%S")
	return datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")

# this function constructs features over a chunk, which contains samples_per_step measurements
# for that time window, for each unprocessed original feature, the min, max, mean, std of the
# samples_per_step is calculated
def define_standard_features(dataset, labels, feature_window):
	print("Constructing features....")
	print("Dataset: ", dataset)
	print("Labels: ", labels)

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


def create_dir(root_dir_path, subdir_name_list=None):
	'''

	Args:
		root_dir_path:
		subdir_name_list:

	Returns:

	'''
	# parent directory
	root_dir = os.path.join(get_project_root_path(), root_dir_path)
	if not os.path.exists(root_dir):
		try:
			os.makedirs(root_dir)
		except OSError as exc:  # Guard against race condition
			if exc.errno != errno.EEXIST:
				raise

	# create subdirectories if there is subdir_name_list passed as a parameter
	if subdir_name_list is not None:
		for subdir_path in subdir_name_list:
			subdir = os.path.join(root_dir, subdir_path)
			if not os.path.exists(subdir):
				try:
					os.makedirs(subdir)
				except OSError as exc:  # Guard against race condition
					if exc.errno != errno.EEXIST:
						raise

	return root_dir

def get_project_root_path():
	return os.path.abspath(os.path.join(os.pardir, os.pardir))