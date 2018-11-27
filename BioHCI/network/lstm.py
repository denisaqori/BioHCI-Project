import torch.nn as nn
from torch.autograd import Variable
import torch


# Some bases from http://pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html
# Also thanks to: https://github.com/yunjey/pytorch-tutorial and https://github.com/MorvanZhou/PyTorch-Tutorial
class LSTM (nn.Module):

	def __init__(self, input_size, hidden_size, output_size, batch_size, batch_first, num_layers, dropout_rate, use_cuda):
		super(LSTM, self).__init__()

		self.name = "LSTM"
		self.hidden_size = hidden_size
		self.output_size = output_size
		self.batch_size = batch_size
		self.batch_first = batch_first
		self.use_cuda = use_cuda

		# the lstm layer that receives inputs of a specific size and outputs
		# a hidden state of hidden _state
		self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
							dropout=dropout_rate, batch_first=batch_first)

		# a linear layer mapping the hidden state to output, then squashing
		# the output (probability for each class) through a softmax function
		self.hidden2out = nn.Linear(hidden_size, output_size)
		# define the softmax function, declaring the dimension along which it will be computed (so every slice along it
		# will sum to 1). The output  (on which the function will be called) will have the shape batch_size x output_size
		self.softmax = nn.LogSoftmax(dim=1)  # already ensured this is the right dimension and calculation is correct

	# The hidden and cell state dimensions are: (num_layers * num-direction, batch, hidden_size) for each
	def init_hidden(self):
		if self.use_cuda:
			# noinspection PyUnresolvedReferences
			return (Variable(torch.zeros(1, self.batch_size, self.hidden_size)).float().cuda(async=True),
					Variable(torch.zeros(1, self.batch_size, self.hidden_size)).float().cuda(async=True))
		else:
			# noinspection PyUnresolvedReferences
			return (Variable(torch.zeros(1, self.batch_size, self.hidden_size)).float(),
					Variable(torch.zeros(1, self.batch_size, self.hidden_size)).float())

	def forward(self, input):
		# (h_0, c_0) assumed zero even if given (but we have defined them above)
		# this is the main layer of the network where all the gradient computation happens
		# the whole sequence is passed through there
		# output contains result for each time step, hidden only for last, and cell state contains the
		# cell state of the last time step

		self.lstm.flatten_parameters()
		# output, (hidden, cell) = self.lstm(input, self.init_hidden())
		output, (hidden, cell) = self.lstm(input, None)

		# the output is returned as (batch_number x sequence_length x hidden_size) since batch_first is set to true
		# otherwise, if the default settings are used, batch number comes second in dimensions
		# we are interested in only the output of the last time step, since this is a many to one network
		if self.batch_first:
			output = self.hidden2out(output[:, -1, :])
		else:
			output = self.hidden2out(output[-1, :, :])

		output = self.softmax(output)
		return output

