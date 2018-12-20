import numpy as np
from copy import copy


class DatasetProcessor:
	def __init__(self, parameters, balancer=None, feature_constructor=None, data_augmenter=None):
		"""
		Args:
			samples_per_chunk (int): an integer indicating how many instances/samples should be in one chunk of data
			balancer:
			interval_overlap (bool): indicates whether the consecutive created intervals/chunks will have any
				overlapping instances. If set to False, instances are simply split into groups of samples_per_step.
				If set to True, additionally new chunks are created based on existing ones, taking the bottom half of
				the instances of the previous chunk, and the top half of those of the next chunk.
		"""
		self.parameters = parameters
		self.balancer = balancer
		self.feature_constructor = feature_constructor
		self.data_augmenter = data_augmenter

		self.data_chunked = False  # used to ensure order of operations (ex: chunking before compacting) for the data
		self.data_compacted = False  # similar to the above (ex. compacting before balancing)

		self.features_constructed = False
		self.data_augmented = False
		self.data_balanced = False

	def chunk_data(self, subj_dict, samples_per_interval, split_axis, interval_overlap):
		"""
		Creates chunks of samples_per_chunk instances, so that the samples input to a classifier or deep neural
		network architecture preserve some timing/continuity information. If at the end of a category there are
		instances that don't add up to a full chunk (made up of samples_per_chunk samples), it pads the chunk with
		zeros at the end.

		Args:
			subj_dict (dict): dictionary mapping subject name to its corresponding Subject object

		Returns:
			chunked_subj_dict (dict): the same dataset represented by the input subj_dict, but each category for each
				subject is 	chunked, so the numpy array corresponding to each category has an extra dimension. The
				data for each Subject object in the returned dictionary is a list of ndarrays of shape:  (number of
				chunks, samples_per_step, number of features)
		"""
		print("\nChunking with interval overlap!!!")

		chunked_subj_dict = {}  # dictionary to return
		# iterate over the subject dictionary and get the corresponding data and category lists
		for subj_name, subject in subj_dict.items():
			subj_data = subject.get_data()
			subj_cat_names = subject.get_categories()

			# create a new list to add chunked category data for the subject
			subj_chunked_data = []
			for i, category in enumerate(subj_data):
				chunked_category = self._chunk_category(category, samples_per_interval, split_axis, interval_overlap)
				subj_chunked_data.append(chunked_category)

			new_subject = copy(subject)  # create a new Subject object with the same values as the original
			new_subject.set_categories(subj_cat_names)  # append the categories
			new_subject.set_data(subj_chunked_data)  # append the new data
			chunked_subj_dict[subj_name] = new_subject  # make this subject the value to the key (name) in dictionary

		self.data_chunked = True
		return chunked_subj_dict

	# TODO: write test cases
	def _chunk_category(self, category, samples_per_interval, split_axis, interval_overlap):
		"""
		Helper function of chunk_data. Chunks the data for one subject's category.

		Args:
			category (2D ndarray): contains data from one category belonging to one subject
			samples_per_chunk (int): an integer indicating how many instances/samples should be in one chunk of data
			interval_overlap (bool): indicates whether the consecutive create intervals/chunks will have any
				overlapping instances. If set to False, instances are simply split into groups of samples_per_step.
				If set to True, besides that, new chunks are created based on existing ones, taking the bottom half
				of the previous chunk, and the top half of instances of the next one.

		Returns:

		"""
		# create list according to which the first dimension of the category numpy array will be split
		assert (split_axis in range(0, len(category.shape) - 1)), "Axis to be split along needs to exist in the " \
																  "category argument."
		assert (samples_per_interval < category.shape[
			split_axis]), "There are not enough instances to make up a chunk in " \
						  "the or if this happens during train-validation split, decrease the number of folds."

		split_list = np.arange(samples_per_interval, category.shape[split_axis], step=samples_per_interval).tolist()

		# split the intervals according to samples_per_chunk with no instances belonging to more than one chunk
		category_chunks = np.split(category, split_list, axis=split_axis)

		# pad the last chunk with zeros if there are not enough instances as for any other chunk (samples_per_step)
		nrows = category_chunks[-1].shape[split_axis]
		print("shape of last category chunk: ", category_chunks[-1].shape)
		print("nrows: ", nrows)
		print("samples per interval: ", samples_per_interval)
		if nrows < 0.5 * samples_per_interval:
			print("Removing last chunk since there are fewer than half of number of instances per interval.")
			category_chunks.pop(-1)
		elif (nrows > 0.5 * samples_per_interval) and (nrows < samples_per_interval):
			print("Padding last chunk since there are more than half of number of instances per interval.")
			rows_to_add = samples_per_interval - nrows
			category_chunks[-1] = np.pad(category_chunks[-1], [(0, rows_to_add), (0, 0)], mode='constant',
									 constant_values=0)

		# if no interval overlap is specified, return the chunked_category. Otherwise, create new chunks based on
		# previous ones.
		if interval_overlap is True:
			# at this point the last array of category chunks has been padded
			category_chunks = self._overlap_intervals(category_chunks)

		# stack the chunks along a new dimension (which will be the axis split_axis (so insert new dim before the one
		#  to split))
		# if interval_overlap is true, category_chunks will contain overlap chunks in addition to the original chunk,
		# otherwise the original ones only
		chunked_category = np.stack(category_chunks, axis=split_axis)
		print("Shape of chunked_category: ", chunked_category.shape)

		return chunked_category

	# TODO: write test cases
	def _overlap_intervals(self, category_chunks):
		"""
		Creates overlapping chunks of data form a list of chunks (2D ndarrays).

		Args:
			category_chunks (list of ndarrays): a list of chunked data, each a 2D ndarry of shape (samples_per_step,
			number of features).

		Returns:
			all_chunks (list of ndarrays): a list of all the original chunks as well as the newly created ones
				based on two adjacent chunks. The total number of elements in the list returned is 2*(number of
				elements in
				the argument list) - 1.

		"""
		# this list will contain original category chunks as well as constructed overlap ones

		all_chunks = [category_chunks[0]]

		for i in range(1, len(category_chunks)):
			half_line = int(category_chunks[i - 1].shape[0] / 2)

			# get the bottom half of the previous chunk
			top_half = category_chunks[i - 1][half_line:]
			# get the top half of the current chunk
			bot_half = category_chunks[i][0:half_line]

			# join the halves to create a new chunk
			new_chunk = np.concatenate((top_half, bot_half), axis=0)

			# add the current chunk and the newly created one into the list to be returned
			all_chunks.append(new_chunk)
			all_chunks.append(category_chunks[i])

		return all_chunks

	# TODO: fix tests to run with compact_subject_categories as a member
	def compact_subject_categories(self, chunked_subj_dict):
		"""
		To be called after the data has been chunked (and padded when necessary). It assumes any
		subject can have data from the same category in different ndarrays, and if that's the case, it recreates the
		subject_dictionary to ensure all the data from one category in one subject is in the same ndarray,
		concatenated across axis=0.

		Args:
			chunked_subj_dict (dict): A dictionary mapping subject names to their corresponding categories

		Returns:
			chunked_subj_dict (dict): A chunked dictionary similar to the input dictionary, but if the subject
				has data from the same category split into different ndarrays, those arrays are concatenated into one
				(across axis=0) in the new Subject object, for each 'subject name' -> Subject object pair of the
				dictionary

		Examples:
		>>> p1_tuple = ([np.array([1, 2, 3]), np.array([4, 5, 6]), np.array([7, 8, 9])], ['a', 'b', 'a'])
		>>> p2_tuple = ([np.array([10, 11, 12]), np.array([13, 14, 15]), np.array([16, 17, 18])], ['b', 'c', 'a'])
		>>> p3_tuple = ([np.array([19, 20, 21]), np.array([22, 23, 24]), np.array([25, 26, 27])], ['b', 'b', 'b'])
		>>> subj_dict = {'p1': p1_tuple, 'p2': p2_tuple, 'p3': p3_tuple}
		>>> DatasetProcessor(subj_dict, 30).compact_subject_categories(subj_dict)['p1']
		([array([1, 2, 3, 7, 8, 9]), array([4, 5, 6])], ['a', 'b'])
		>>> DatasetProcessor(self.subj_dict, self.samples_per_chunk).compact_subject_categories(subj_dict)['p2']
		([array([10, 11, 12]), array([13, 14, 15]), array([16, 17, 18])], ['b', 'c', 'a'])
		>>> compact_subject_categories({'p1': p1_tuple, 'p2': p2_tuple, 'p3': p3_tuple})['p3']
		([array([19, 20, 21, 22, 23, 24, 25, 26, 27])], ['b'])
		"""

		assert self.data_chunked is True, "Data has not been chunked in DataSplitter, so no interval overlap is " \
										  "possible. Frist call <instance of DataSplitter>.chunk_data(subj_dict, " \
										  "samples_per_chunk, interval_overlap), then call this method again.)"

		# dictionary to return
		compacted_subj_dict = {}
		# iterate over the original dictionary and for each subject
		for subj_name, subject in chunked_subj_dict.items():
			# extract the list of data split according to categories
			subj_cat_data = subject.get_data()
			# as well as the list of corresponding (by index) category names
			subj_cat = subject.get_categories()
			# get unique category names
			cat_set = set(subj_cat)

			# if the category names in the list are not unique
			if len(cat_set) != len(subj_cat):
				# calculate the indices each category maps to
				val_to_ind = self.find_indices_of_duplicates(subj_cat)
				# iterate over the category names and corresponding indices
				new_subj_cat_names = []
				compact_data_list = []
				for cat_name, index_list in val_to_ind.items():
					compact_data = subj_cat_data[index_list[0]]
					new_subj_cat_names.append(cat_name)

					# concatenate data from the same category along axis = 0
					if len(index_list) > 1:
						for i in range(1, len(index_list)):
							print("i = ", i)
							new_data = subj_cat_data[index_list[i]]
							compact_data = np.concatenate((compact_data, new_data), axis=0)
					# the list of compacted data to be associated with a subject
					compact_data_list.append(compact_data)

				new_subject = copy(subject)
				new_subject.set_data(compact_data_list)
				new_subject.set_categories(new_subj_cat_names)
				compacted_subj_dict[subj_name] = subject
			else:
				compacted_subj_dict[subj_name] = subject

		self.data_compacted = True
		return compacted_subj_dict

	# To run doctests on any function/method of this module, open the terminal and type:
	# python -m doctest -v data_splitter.py
	def find_indices_of_duplicates(self, ls):
		"""
		Calculates the indices of every unique value of the list.

		Args:
			ls (list): a list of values (strings)

		Returns:
			ordered_dict (dictionary): a dictionary sorted by key, mapping every unique value of the list, to the list
			of the indices at which that value is found in ls.

		Examples:
			>>> find_indices_of_duplicates(['a', 'b', 'c', 'd', 'a', 'd'])['a']
			[0, 4]
			>>> find_indices_of_duplicates(['a', 'b', 'c', 'd', 'a', 'd'])['c']
			[2]
		"""
		# create a set of all list elements to ensure that it has only the unique values of the list
		elem_set = set(ls)
		# dictionary to return
		name_to_indices = {}

		# for each unique list element
		for set_elem in elem_set:
			same_elem = []
			# find the index/indices that belong to it in the original list
			for i, list_elem in enumerate(ls):
				if set_elem == list_elem:
					same_elem.append(i)
			# assign the list of its indices to the list value
			name_to_indices[set_elem] = same_elem

		return name_to_indices

	def construct_features(self, compacted_subj_dict):
		"""
		Constructs features on the samples in the given training or testing set (dictionary) so that the instances of
		each category are collapsed to create several representative features.

		Args:
			compacted_subj_dict (dict): A dictionary mapping subject names to the corresponding Subject object.
				For each Subject object, the data from each category is in one ndarray only, and each category appears
				exactly once. The data list corresponds to the category list.

		Returns:
			feature_dataset(dict): A dictionary similar to the above, where features over some interval are built.

		"""
		assert self.data_compacted is True, "Data has not been compacted in DatasetProcessor, so data from one " \
											"category may not be uniquely able to be indexed. First call " \
											"compact_subject_categorie(chunked_subj_dict), then call this method " \
											"again."

		assert self.data_chunked is True, "Data has not been chunked in DatasetProcessor. First call chunk_data(" \
										  "subj_dict, samples_per_chunk, interval_overlap), then call this " \
										  "method again."

		if self.feature_constructor is None:
			print("No feature_constructor argument set in the DatasetProcessor Object. Skipping this step "
				  "and returning original dictionary (passed as an argument to this funtion) which has been chunked "
				  "and compacted...")
			return compacted_subj_dict

		else:
			# feature_dataset = self.feature_constructor.construct_features(compacted_subj_dict)
			feature_dataset = self.chunk_data(compacted_subj_dict, self.parameters.feature_window, 1,
											  self.parameters.feature_overlap)
			feature_dataset = self.feature_constructor.get_stat_features(feature_dataset, feature_axis=2)

			print("Returning category-balanced dictionary...")
			return feature_dataset

	def balance_categories(self, compacted_subj_dict):
		"""
		Balances the samples in the given training or testing set (dictionary) so that each category has the same
		number of samples. It uses the internal CategoryBalancer strategy to determine the type of balancing applied
		to the dataset.

		Args:
			compacted_subj_dict (dict): A dictionary mapping subject names to the corresponding Subject object.
				For each Subject object, the data from each category is in one ndarray only, and each category appears
				exactly once. The data list corresponds to the category list.

		Returns:
			balanced_dictionary (dict): A dictionary similar to the above, where each category has the same number of
				samples.

		"""
		assert self.data_compacted is True, "Data has not been compacted in DatasetProcessor, so data from one " \
											"category may not be uniquely able to be indexed. First call " \
											"compact_subject_categorie(chunked_subj_dict), then call this method " \
											"again."

		assert self.data_chunked is True, "Data has not been chunked in DatasetProcessor. First call chunk_data(" \
										  "subj_dict, samples_per_chunk, interval_overlap), then call this " \
										  "method again."

		if self.balancer is None:
			print("No category balancer (balancer) argument set in the DatasetProcessor Object. Skipping this step "
				  "and returning original dictionary (passed as an argument to this funtion) which has been chunked "
				  "and compacted...")
			return compacted_subj_dict

		else:
			balanced_dictionary = self.balancer.balance(compacted_subj_dict)
			print("Returning category-balanced dictionary...")
			return balanced_dictionary

	def set_category_balancer(self, balancer):
		"""
		Sets the category balancer to

		Args:
			balancer: a subclass of abstract CategoryBalancer strategy

		"""

		self.balancer = balancer

	def set_feature_constructor(self, feature_constructor):
		self.feature_constructor = feature_constructor

	def set_data_augmenter(self, data_augmenter):
		self.data_augmenter = data_augmenter

	# TODO: this variable checks are not fool-proof, they are easy to break if we call the below functions
	# individually on two different datasets out of order
	def process_dataset(self, subject_dictionary):

		# TODO: maybe feature construction and data augmentation before balancing
		# use the built-in variables to ensure order: 1) chunking 2) compacting 3) balancing
		chunked_subj_dict = self.chunk_data(subject_dictionary, self.parameters.samples_per_chunk, 0,
											self.parameters.interval_overlap)
		compacted_data = self.compact_subject_categories(chunked_subj_dict)
		feature_dataset = self.construct_features(compacted_data)
		balanced_dataset = self.balance_categories(feature_dataset)

		# reset these internal variables so that the same data_processor object can be used on more than one dataset
		self.reset_order_booleans()

		return balanced_dataset

	def reset_order_booleans(self):
		self.data_chunked = False
		self.data_compacted = False
		self.data_augmented = False
		self.features_constructed = False
