import torch.nn as nn
import torch

class CNN (nn.Module):
	def __init__(self, input_size, hidden_size, output_size):
		super(CNN, self).__init__()

		self.name = "CNN"
		self.hidden_size = hidden_size

		self.conv1 = nn.Sequential(		# data_chunk_tensor has shape: (batch_size x samples_per_step x num_attr)
										# the actual input is (samples_per_step x num_attr), batch_size is implicit
			nn.Conv1d(
				in_channels = input_size,
				out_channels = 16,		# number of filters
				kernel_size = 5,        # size of filter
				stride = 1,             # filter movement/step
				padding = 0				# padding=(kernel_size-1)/2 if stride=1
			),
			nn.ReLU(),
			nn.MaxPool1d (kernel_size=2)
		)

		self.conv2 = nn.Sequential (
			nn.Conv1d(
				in_channels = 16,	# input based on the output of the previous layer
				out_channels = 32,  # output channel number
				kernel_size = 5,
				stride = 1,
				padding = 0
			),
			nn.ReLU(),
			nn.MaxPool1d (kernel_size=2)
		)

		self.linear = nn.Linear(32 * 4, output_size) # the first argument is based on the size of the output of conv2
													# and is of size output.size(0) * output.size(1)

	def forward(self, input):
		# print ("Input before transpose: ", input)
		# reshape the input from (batch_size x seq_len x input_size) to (batch_size x input_size x seq_len)
		# since that is how CNN expects it
		input = torch.transpose(input, 1, 2)
		print ("Input to cnn before conv1:", input)

		conv1_parameters = list(self.conv1.parameters())
		print ("Conv1 kernels initially: ", conv1_parameters[0])
		print ("Conv1 biases initially: ", conv1_parameters[1])


		input = self.conv1(input)
#		print ("Input after cnn1 and before cnn2:", input)
		input = self.conv2(input)
#		print ("Input after conv2:", input)

		# flatten input to the linear layer such that it has shape (batch_size x output.size(0)*output.size(1)*output.size(2)
		input = input.view(input.size(0), -1)
#		print ("Input before linear transformation: ", input)
		output = self.linear(input)
		print ("Output of the forward step: ", output)

		return output