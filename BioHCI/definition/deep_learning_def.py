from BioHCI.network.cnn_lstm import CNN_LSTM
from BioHCI.network.cnn import CNN
from BioHCI.network.lstm import LSTM
from BioHCI.definition.learning_def import LearningDefinition
import torch
import torch.nn as nn
import sys


class DeepLearningDefinition(LearningDefinition):
	def __init__(self, model_name, num_features, output_size, use_cuda):
		# hyper-parameters
		self.__num_hidden = 120  # number of nodes per hidden layer
		self.__num_epochs = 20  # number of epochs over which to train
		self.__samples_per_chunk = 25  # the number of measurements to be included in one sequence
		self.__learning_rate = 0.1  # If you set this too high, it might explode. If too low, it might not learn
		self._batch_size = 1  # The number of instances in one batch
		self.__dropout_rate = 0.05  # dropout rate: if 0, no dropout - to be passed to the network model
		self.__num_layers = 2  # number of layers of LSTM
		self.__batch_first = True

		# DOUBLE CHECK THE NEED TO ASSIGN THESE ATTRIBUTES. GETTERS?
		# parameters passed to this object - there should be no setters for these attributes
		self._use_cuda = use_cuda
		self.__num_features = num_features
		self.__output_size = output_size

		# initialize the network, pick the optimizer and the loss function
		# in each case batch_size is set to true, so that input and output are expected to have the batch number as
		# the first dimension (dim=0) instead of it being the second one (dim=1) which is the default

		self._all_train_losses = []

		super(DeepLearningDefinition, self).__init__(model_name)

		# model definition
		# self.model = self.__build_model(self.model_name)

		# the stochastic gradient descent function to update weights and biases
		self.__optimizer = torch.optim.Adam(self.model.parameters(), lr=self.__learning_rate)

		# the negative log likelihood loss function - useful to train classification problems with C classes
		self.__criterion = nn.NLLLoss()

	# getters - the only way to access the class attributes
	def get_num_hidden(self):
		return self.__num_hidden

	def get_num_epochs(self):
		return self.__num_epochs

	def get_samples_per_chunk(self):
		return self.__samples_per_chunk

	def get_batch_size(self):
		return self._batch_size

	def is_batch_first(self):
		return self.__batch_first

	def get_learning_rate(self):
		return self.__learning_rate

	def get_dropout_rate(self):
		return self.__dropout_rate

	def get_num_layers(self):
		return self.__num_layers

# TODO: factory method
	def _build_model(self, name):
		if name == "CNN":
			model = CNN(input_size=self.__num_features, hidden_size=self.__num_hidden,
						 output_size=self.__output_size)
		elif name == "LSTM":
			model = LSTM(input_size=self.__num_features, hidden_size=self.__num_hidden, output_size=self.__output_size,
						batch_size=self._batch_size, batch_first=self.__batch_first, num_layers=self.__num_layers,
						 dropout_rate=self.__dropout_rate, use_cuda=self._use_cuda)
		elif name == "CNN_LSTM":
			model = CNN_LSTM(input_size=self.__num_features, hidden_size=self.__num_hidden,
								output_size=self.__output_size, batch_size=self._batch_size,
							 batch_first=self.__batch_first,
								num_layers=self.__num_layers, dropout_rate=self.__dropout_rate,
								use_cuda=self._use_cuda)
		else:
			print("Model specified in DeepLearningDefinition object is currently undefined!")
			sys.exit()

		return model

	def get_optimizer(self):
		return self.__optimizer

	def get_criterion(self):
		return self.__criterion

	def is_use_cuda(self):
		return self._use_cuda

	def get_num_features(self):
		return self.__num_features

	def get_output_size(self):
		return self.__output_size

	# setters - to be used by a UI; does not include arguments with which the object is created
	def set_num_hidden(self, num_hidden):
		self.__num_hidden = num_hidden

	def set_num_epochs(self, num_epochs):
		self.__num_epochs = num_epochs

	def set_samples_per_chunk(self, samples_per_chunk):
		self.__samples_per_chunk = samples_per_chunk

	def set_batch_size(self, batch_size):
		self._batch_size = batch_size

	def set_bool_batch_first(self, batch_first):
		self.__batch_first = batch_first

	def set_model_name(self, model_name):
		self.model_name = model_name

	def set_learning_rate(self, learning_rate):
		self.__learning_rate = learning_rate

	def set_dropout_rate(self, dropout_rate):
		self.__dropout_rate = dropout_rate

	def set_num_layers(self, num_layers):
		self.__num_layers = num_layers

	def set_model(self, model):
		self.model = model

	def set_optimizer(self, optimizer):
		self.__optimizer = optimizer

	def set_criterion(self, criterion):
		self.__criterion = criterion
