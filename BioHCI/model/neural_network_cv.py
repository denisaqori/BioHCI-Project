from BioHCI.model.cross_validator import CrossValidation
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from BioHCI.model.trainer import Trainer
from BioHCI.model.evaluator import Evaluator
import torch
import os


class NeuralNetworkCV(CrossValidation):

	def __init__(self, subject_dict, data_splitter, dataset_processor, parameter, learning_def,
				 num_categories):

		# this list contains lists of accuracies for each epoch. There will be self._num_folds lists of _num_epochs
		# elements in this list after all training is done
		self._all_epoch_train_accuracies = []

		super(NeuralNetworkCV, self).__init__(subject_dict, data_splitter, dataset_processor, parameter, learning_def,
											  num_categories)
		assert (parameter.neural_net is True), "In StudyParameters, neural_net is set to False and you are " \
													   "trying to instantiate a NeuralNetworkCV object!"

	# implement the abstract method from the parent class CrossValidation; returns a dataset with labels wrapped in
	# the PyTorch DataLoader format
	def _get_data_and_labels(self, python_dataset):
		data, labels = self._data_processor.get_shuffled_dataset_and_labels(python_dataset)

		# the tensor_dataset is a tuple of TensorDataset type, containing a tensor with data (train or val),
		# and one with labels (train or val respectively)
		tensor_dataset = TensorDataset(data, labels)

		print("Using the PyTorch DataLoader to load the training data (shuffled) with: \nbatch size = ",
			  self._learning_def.get_batch_size(), " & number of threads = ", self._parameter.get_num_threads())
		data_loader = DataLoader(tensor_dataset, batch_size=self._learning_def.get_batch_size(),
								  num_workers=self._parameter.get_num_threads(), shuffle=True, pin_memory=True)

		return data_loader

	# implement the abstract method from the parent class CrossValidation; it is called for each fold in
	# cross-validation and after it trains for that fold, it appends the calculated losses and accuracies for each epoch
	# to the respective list in the CrossValidation object
	def train(self, train_dataset):

		trainer = Trainer(train_dataset, self._data, self._learning_def)

		# get the loss over all epochs for this cv-fold and append it to the list
		self._all_train_losses.append(trainer.get_epoch_losses())
		print("Train Epoch Losses: ", trainer.get_epoch_losses())

		# accuracies for each epoch and each fold are added to the list that belongs only to this class
		# "_all_epoch_train_accuracies". The last accuracy of each train epoch is added to the list
		# "_all_train_accuracies, belonging more generally to the parent class
		self._all_train_accuracies.append(trainer.get_epoch_accuracies()[-1])
		self._all_epoch_train_accuracies.append(trainer.get_epoch_accuracies())
		print("Train Epoch Accuracies: ", trainer.get_epoch_accuracies())

	# evaluate the model created during training on the valing dataset
	def val(self, val_dataset):
		# this is the network produces by training over the other folds
		model_name = self._data.get_dataset_name() + "-" + self._learning_def.get_model_name() + "-batch-" \
					+ str(self._learning_def.get_batch_size()) + "-seqSize-" \
					+ str(self._learning_def.get_samples_per_step()) + ".pt"

		model_to_eval = torch.load(os.path.join("saved_models", model_name))

		evaluator = Evaluator(test_data_loader=val_dataset, model_to_eval=model_to_eval,
							  categories=self._data.get_categories(),
							  confusion=self._confusion, neural_network_def=self._learning_def)

		fold_accuracy = evaluator.get_accuracy()
		self._all_val_accuracies.append(fold_accuracy)

	def get_all_epoch_train_accuracies(self):
		return self._all_epoch_train_accuracies