from BioHCI.model.cross_validator import CrossValidator
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from BioHCI.model.trainer import Trainer
from BioHCI.model.evaluator import Evaluator
from BioHCI.helpers import utilities as util

import torch.nn as nn
import torch
import os


class NNCrossValidator(CrossValidator):

	def __init__(self, subject_dict, data_splitter, feature_constructor, model, parameter,
				 learning_def, all_categories):
		# this list contains lists of accuracies for each epoch. There will be self._num_folds lists of _num_epochs
		# elements in this list after all training is done
		self._all_epoch_train_accuracies = []
		self.parameters = parameter
		self.model = model

		# the stochastic gradient descent function to update weights and biases
		self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_def.learning_rate)

		# the negative log likelihood loss function - useful to train classification problems with C classes
		self.criterion = nn.NLLLoss()

		super(NNCrossValidator, self).__init__(subject_dict, data_splitter, feature_constructor,
											   model, parameter, learning_def, all_categories)
		assert (parameter.neural_net is True), "In StudyParameters, neural_net is set to False and you are " \
											   "trying to instantiate a NNCrossValidator object!"

	# implement the abstract method from the parent class CrossValidator; returns a dataset with labels wrapped in
	# the PyTorch DataLoader format
	def _get_data_and_labels(self, subj_dataset):
		# data, labels = self._data_processor.get_shuffled_dataset_and_labels(python_dataset)

		data, cat = self.mix_subj_chunks(subj_dataset)

		# convert numpy ndarray to PyTorch tensor
		data = torch.from_numpy(data)
		# convert categories from string to integer
		int_cat = self.convert_categories(cat)
		cat = torch.from_numpy(int_cat)

		# the tensor_dataset is a tuple of TensorDataset type, containing a tensor with data (train or val),
		# and one with labels (train or val respectively)
		tensor_dataset = TensorDataset(data, cat)

		print("Using the PyTorch DataLoader to load the training data (shuffled) with: \nbatch size = ",
			  self._learning_def.batch_size, " & number of threads = ", self._parameter.num_threads)
		data_loader = DataLoader(tensor_dataset, batch_size=self._learning_def.batch_size,
								 num_workers=self._parameter.num_threads, shuffle=False, pin_memory=True)

		return data_loader

	# implement the abstract method from the parent class CrossValidator; it is called for each fold in
	# cross-validation and after it trains for that fold, it appends the calculated losses and accuracies for each
	# epoch to the respective list in the CrossValidator object
	def train(self, train_dataset, summary_writer):
		train_data_loader = self._get_data_and_labels(train_dataset)
		trainer = Trainer(train_data_loader, self.model, self.optimizer, self.criterion, self.all_int_categories,
						  self._learning_def, self.parameters, summary_writer)

		# get the loss over all epochs for this cv-fold and append it to the list
		self._all_train_losses.append(trainer.get_epoch_losses())
		print("Train Epoch Losses: ", trainer.get_epoch_losses())

		# accuracies for each epoch and each fold are added to the list that belongs only to this class
		# "_all_epoch_train_accuracies". The last accuracy of each train epoch is added to the list
		# "_all_train_accuracies, belonging more generally to the parent class
		self._all_train_accuracies.append(trainer.get_epoch_accuracies()[-1])
		self._all_epoch_train_accuracies.append(trainer.get_epoch_accuracies())
		print("Train Epoch Accuracies: ", trainer.get_epoch_accuracies())

	# evaluate the model created during training on the validation dataset
	def val(self, val_dataset, summary_writer):
		val_data_loader = self._get_data_and_labels(val_dataset)

		# this is the network produces by training over the other folds
		model_name = self._parameter.study_name + "-" + self.model.name + "-batch-" \
					 + str(self._learning_def.batch_size) + "-seqSize-" \
					 + str(self.parameters.samples_per_chunk) + ".pt"

		saved_models_root = util.get_root_path("saved_models")
		model_to_eval = torch.load(os.path.join(saved_models_root, model_name))

		evaluator = Evaluator(val_data_loader, model_to_eval, self.all_int_categories, self._confusion,
							  self._learning_def, summary_writer)

		fold_accuracy = evaluator.get_accuracy()
		self._all_val_accuracies.append(fold_accuracy)

	def get_all_epoch_train_accuracies(self):
		return self._all_epoch_train_accuracies
