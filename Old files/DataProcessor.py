import torch
import random


class DataProcessor:
	def __init__(self, data, samples_per_step):
		print("\nFurther Data Processing...\n")

		self.data = data
		self.samples_per_step = samples_per_step

		self.dataset = self.data.get_dataset()
		self.categories = self.data.categories

		self.tensor_dataset = self.to_tensor_chunked(self.dataset)

	# this method takes as argument a dataset, a python list which contains all the data for each category in numpy
	# format with each category having the shape (num_subj x inst_per_subj x num_attr) the instances (axis 1) are
	# split into chunks of size samples_per_step; returns a python list with the category data in torch format,
	# with an extra dimension to account for the split; the shape of the returned tensor dataset, a python list with
	# data from each category is (num_subj x samples_per_step x num_attr x num_chunks)
	# TODO: introduce overlap of instances per chunk - more information retention?
	# IMPORTANT: needs to be done after test/train set separation
	def to_tensor_chunked(self, dataset):

		tensor_dataset = []
		for i, category_data in enumerate(dataset):
			tensor_data = torch.from_numpy(category_data)
			tensor_data_split = torch.split(tensor_data, self.samples_per_step, dim=1)
			print(
				"The data from each file in category ", self.categories[i], " will be split into : ",
				len(tensor_data_split), " chunks of ", self.samples_per_step, " instances.")
			print("TensorDataSplit: \n", tensor_data_split[0])

			# concatenate along new dimension inserted at the end
			tensor_data = torch.stack(tensor_data_split, dim=3)

			print("Size of new tensor data: \n", tensor_data.size())
			tensor_dataset.append(tensor_data)
			print("TensorData [0,:,:,0]: \n", tensor_data[0, :, :, 0])

		return tensor_dataset

	# this method splits the data into k folds across the axis=-1 (or axis 3 - num_chunks) so that samples_per_step
	# dimension stays consecutive at this point it selects fold s to test on and uses the rest as training data #
	# TODO: split data into train:test:validation in the ratio: 70:20:10 - test set should not be used when selecting parameters
	# TODO: have the option of one participant's data not being used across the three sets, but only in one
	def split_in_folds(self, k=10, s=0):
		print("Starting the dataset split for cross validation, with k =", k, "along axis 3",
			  "which means there will be", (self.tensor_dataset[0].size()[-1]) / k, "chunks of",
			  self.samples_per_step,
			  "from each file/subject for each category.")
		print("The fold selected for testing in this instance is s =", s)

		# this list will contain the folds for each category, with category_folds[x]
		# containing the split data in k sets of category[x] along axis=-1 (3)
		# the first sub-array of category_folds[x] will contain the first part of the
		# data, and so on for the rest
		category_folds = []

		# splitting each category data into k groups along axis 3
		for i, category_data in enumerate(self.tensor_dataset):
			# the folds for each category
			cat_fold = torch.chunk(category_data, k, dim=3)
			category_folds.append(cat_fold)

		# printing the shape of each fold in each category
		for i, category in enumerate(category_folds):
			for j, folds in enumerate(category):
				print("Category ", self.categories[i], ": ", folds.shape)

		# these lists will be two datasets, with each containing at index x a tensor
		# of each category
		train = []
		test = []

		# there will be one fold for testing and the rest will be used for training
		# for each category; here there is a tensor for each category
		# to be inserted in the test[] test at the specified index, while the rest
		# needs to be eventually inserted in the train set -> for now just in train_folds
		train_folds = []
		for i, cat_folds in enumerate(category_folds):
			train_cat = []
			for j in range(len(cat_folds)):
				fold = cat_folds[j]
				if j == s:
					test.append(fold)
				else:
					train_cat.append(fold)
			train_folds.append(train_cat)

		# create a tensor with the folds for training for each category
		# and insert it in the train[] list - we concatenate along the last dimension (3)
		# which means we add the chunks on top of each other
		for i, cat_folds in enumerate(train_folds):
			cat_train = torch.DoubleTensor(cat_folds[0])

			for j in range(1, (k - 1)):
				cat_train = torch.cat((cat_train, cat_folds[j]), dim=3)

			train.append(cat_train)

		for i, train_data in enumerate(train):
			print("Shape of train data for category ", self.categories[i], ": ", train[i].size())

		for i, test_data in enumerate(test):
			print("Shape of test data for category ", self.categories[i], ": ", test[i].size())

		# ensure or create balanced testing and training datasets - the same number of instances for all classes
		print("\nTESTING SET")
		if self.is_balanced(test):
			balanced_test = test
		else:
			balanced_test = self.balance_classes(test, 'Oversample')

		print("\nTRAINING SET")
		if self.is_balanced(train):
			balanced_train = train
		else:
			balanced_train = self.balance_classes(train, 'Oversample')

		return balanced_train, balanced_test

	# this method checks to see if the data is balanced for each class. The format of the dataset variable is assumed
	# to be a python list of tensors of shape ((number of subjects in category) x (number of instances per step) x (
	# number of features) x (number of chunks per file/subj)) the number of features and number of samples/instances
	# per step/chunk are constant the only differing axis are number of subjects of a particular label, and number of
	#  chunks
	def is_balanced(self, dataset):
		print("\nChecking whether classes are balanced: ")
		balanced = True
		for i in range(len(dataset)):
			print("Category: ", self.categories[i], " Number of files/subjects: ", dataset[i].size()[0],
				  "Number of total chunks: ", dataset[i].size()[0] * dataset[i].size()[3])

			if dataset[0].size()[0] * dataset[0].size()[3] != dataset[i].size()[0] * dataset[i].size()[3]:
				balanced = False

		print("\n")
		for k, category_data in enumerate(dataset):
			print("Category ", self.categories[k], " data shape: ", category_data.size())

		print("\nClasses are balanced: ", balanced)
		return balanced

	# this method oversamples the class that has fewer instances so that there is
	# an equal number of instances for each. The format of the dataset variable is assumed to be the
	# same as that of index_labeled_dataset: a python list of numpy arrays of shape
	# ((number of subjects in category) x (number of instances per file/subject) x (number of features))
	def balance_classes(self, dataset, balance_type="Oversample"):

		# the dataset that will be eventually returned with balanced classes
		# (dataset[x].shape[0] * dataset[x].shape[1]) will be the same for every element of the list balanced_dataset
		balanced_dataset = []

		# this list contains the total instance count for each class (dataset[x].shape[0] * dataset[x].shape[1])
		class_chunk_count = []
		if balance_type == 'Oversample':
			print("\nBalancing classes by oversampling...")
			for j, category_data in enumerate(dataset):
				class_chunk_count.append(dataset[j].size()[0] * dataset[j].size()[-1])

			# the variable max contains the maximum value of the list of number of instances
			max = 0
			for val in class_chunk_count:
				if val > max:
					max = val
			print("\nThe categories/classes: ", self.categories)
			print("The number of total chunks of instances/samples per class before balancing: ", class_chunk_count)
			print("The maximum number of instances is: ", max)

			# find the number of instances that needs to be added to each class
			diff = [max - val for val in class_chunk_count]
			print("The number of chunks of instances/samples that need to be added to each category: ", diff)

			# the final dataset to be returned with balanced categories
			balanced_dataset = []
			# determine the number of instances to oversample per file/subject per class so each contributes
			# equally to the new instance set.
			for k, category_data in enumerate(dataset):
				balanced_cat_chunks = []
				for l in range(category_data.size()[0]):
					# num_to_add contains the number of chunks of instances to be sampled from each file of the same
					# class. The if/else clause serves to account for diff[x] () not being divisible by the number of
					# subjects/files - we add an extra chunk for a certain number of files (remainder of diff[
					# x]/number of files/subjects)
					if l < (diff[k] % dataset[k].size()[0]):
						num_to_add = 1 + int(diff[k] / dataset[k].size()[0])
					else:
						num_to_add = int(diff[k] / dataset[k].size()[0])

					print("Number of chunks to add per file/subject: ", num_to_add)

					# idx is an array of size=num_to_add, containing the indices to duplicate out of
					# all the possible indices of category_data.size()[3] (number of chunks of instances)
					# the chosen indices are unique, so that the same value is not added twice to the set
					idx = random.sample(range(0, int(category_data.size()[-1])), k=num_to_add)
					extra_idx = torch.LongTensor(idx)

					# to each subject data we add more instances if we are to oversample it
					if num_to_add > 0:
						# extra instances to be added to the specific category in the last axis
						# for each file/subject of the specific category
						extra_chunks = torch.index_select(category_data[l, :, :, :], -1, extra_idx)
						subj_data = torch.cat((category_data[l, :, :, :], extra_chunks), dim=-1)
					else:
						subj_data = category_data[l, :, :, :]

					# appending the subject data to the list with other subj. of the same category
					balanced_cat_chunks.append(subj_data)

				# stack the data from every subject and create the balanced category by adding dimension 0
				# then append the newly created set of instances to the balanced_dataset list
				chunks_in_tuple = tuple(balanced_cat_chunks[i] for i in range(category_data.size()[0]))
				balanced_category = torch.stack(chunks_in_tuple, dim=0)
				balanced_dataset.append(balanced_category)

				print("Shape of balanced category ", self.categories[k], ": ", balanced_category.size(), "\n")

		return balanced_dataset

	# this function takes in a dataset (python list of 4D tensors) and returns 2 tensors, one of the data and the
	# other of its corresponding labels, both shuffled together; The returned dataset tensor has the shape (number of
	# total chunks) x (number of instances per chunk) x (number of features) - the labels tensor has only one
	# dimension, that of the number of total chunks, which is the same as the number of total labels returned
	def get_shuffled_dataset_and_labels(self, dataset):
		labels_tuple = ()
		categories_tuple = ()
		for k, category_data in enumerate(dataset):
			print("Reshaping category data to so all chunks from all files are in the same dimension (dim=0)")
			print("Category shape: ", category_data.size())
			category_data = category_data.contiguous()
			new_category_view = category_data.view(category_data.size()[0] * category_data.size()[-1],
												   category_data.size()[1], category_data.size()[2])
			print("New category ", self.categories[k], "view: ", new_category_view.size())
			categories_tuple = categories_tuple + (new_category_view,)

			# assign a number to each category (incrementally)
			cat_labels = k * torch.ones(new_category_view.size()[0])
			print("Category Labels: ", cat_labels)
			labels_tuple = labels_tuple + (cat_labels,)

		# concatenate category labels
		labels = torch.cat(labels_tuple)

		# concatenate category data - corresponding to the above labels
		dataset = torch.cat((categories_tuple), dim=0)

		print("Dataset: ", dataset)
		print("Labels: ", labels)

		return dataset, labels
