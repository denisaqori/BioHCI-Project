import torch.nn as nn
from torch.autograd import Variable
import torch


class CNN_LSTM (nn.Module):
	def __init__(self, input_size, hidden_size, output_size, batch_size, batch_first, num_layers, dropout_rate, use_cuda):
		super(CNN_LSTM, self).__init__()

		self.name = "CNN_LSTM"
		self.hidden_size = hidden_size
		self.use_cuda = use_cuda
		self.batch_size = batch_size
		self.batch_first = batch_first

		self.conv1 = nn.Sequential(		# data_chunk_tensor has shape: (batch_size x samples_per_step x num_features)
			# the actual input is (samples_per_step x num_features), batch_size is implicit
			# input size expected by Conv1d: (batch_size x number of channels x length of signal sequence)
			# so in our case it needs to be (batch_size x input_size x samples_per_step)
			nn.Conv1d(
				in_channels = input_size,
				out_channels = 16,		# number of filters
				kernel_size = 5,         # size of filter
				stride = 1,             # filter movement/step
				padding = 2				# padding=(kernel_size-1)/2 if stride=1 -> added to both sides of input
			),
			nn.ReLU(),
			nn.MaxPool1d(kernel_size=2)
		)

		self.lstm = nn.LSTM(input_size=16, hidden_size=hidden_size, num_layers=num_layers,
							dropout=dropout_rate, batch_first=batch_first)

		self.hidden2out = nn.Linear(hidden_size, output_size)
		self.softmax = nn.LogSoftmax(dim=1) # already ensured this is the right dimension and calculation is correct

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
		# print("Input before transpose: ", input.size())
		# reshape the input from (batch_size x seq_len x input_size) to (batch_size x input_size x seq_len)
		# since that is how CNN expects it

		# noinspection PyUnresolvedReferences
		input = torch.transpose(input, 1, 2)
		# print("Input to cnn before conv1:", input.size())
		input = self.conv1(input)
		# the output of conv1d is expected to be (batch_size x output_channels(number of kernels) x seq_len)
		# but seq_len can be shorter, since it's valid cross-correlation not full cross-correlation
		# print("Input after conv1:", input.size())

		# transpose input again since LSTM expects it as (batch_size x seq_len x input_size)
		# noinspection PyUnresolvedReferences
		input = torch.transpose(input, 1, 2)
		# print("Input after transpose before lstm:", input.size())

		self.lstm.flatten_parameters()
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