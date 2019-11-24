import torch
import torch.nn as nn
from torch.autograd import Variable

from BioHCI.architectures.abstract_neural_net import AbstractNeuralNetwork


class CNN2D_LSTM_C(AbstractNeuralNetwork):

    def __init__(self, nn_learning_def):
        super(CNN2D_LSTM_C, self).__init__()

        self.__name = "CNN2D_LSTM_classification"
        self.hidden_size = nn_learning_def.num_hidden
        self.use_cuda = nn_learning_def.use_cuda
        self.batch_size = nn_learning_def.batch_size
        self.batch_first = nn_learning_def.batch_first
        self.input_size = nn_learning_def.input_size
        self.num_layers = nn_learning_def.num_layers
        self.dropout_rate = nn_learning_def.dropout_rate
        self.output_size = nn_learning_def.output_size

        self.conv = nn.Sequential(  # data_chunk_tensor has shape: (batch_size x samples_per_chunk x num_attr)
            # the actual input is (samples_per_chunk x num_attr), batch_size is implicit
            # input size expected by Conv1d: (batch_size x number of channels x length of signal sequence)
            # so in our case it needs to be (batch_size x input_size x samples_per_chunk)
            nn.Conv2d(
                in_channels=self.input_size,
                out_channels=self.input_size*8,  # number of filters
                kernel_size=5,  # size of filter
                stride=1,  # filter movement/step
                padding=2  # padding=(kernel_size-1)/2 if stride=1 -> added to both sides of input
                ),
            nn.ReLU(),
            nn.BatchNorm2d(self.input_size*8),
            # nn.Dropout(),
            nn.Conv2d(
                 in_channels=self.input_size*8,
                 out_channels=32,  # number of filters
                 kernel_size=5,  # size of filter
                 stride=1,  # filter movement/step
                 padding=2  # padding=(kernel_size-1)/2 if willstride=1 -> added to both sides of input
                ),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=2)
            )

        # xavier initialization for convolutional layer
        self.conv.apply(self.weights_init)

        self.lstm = nn.LSTM(input_size=32, hidden_size=self.hidden_size, num_layers=self.num_layers,
                            dropout=self.dropout_rate, batch_first=self.batch_first)

        self.hidden = nn.Linear(32*500, self.hidden_size)
        self.hidden2out = nn.Linear(self.hidden_size, self.output_size)
        self.softmax = nn.LogSoftmax(dim=1)  # already ensured this is the right dimension and calculation is correct

    def weights_init(self, m):
        if isinstance(m, nn.Conv1d):
            torch.nn.init.xavier_uniform(m.weight.data)

    def init_hidden(self):
        if self.use_cuda:
            # noinspection PyUnresolvedReferences
            return (Variable(torch.zeros(1, self.batch_size, self.hidden_size)).float().cuda(),
                    Variable(torch.zeros(1, self.batch_size, self.hidden_size)).float().cuda())
        else:
            # noinspection PyUnresolvedReferences
            return (Variable(torch.zeros(1, self.batch_size, self.hidden_size)).float(),
                    Variable(torch.zeros(1, self.batch_size, self.hidden_size)).float())

    def forward(self, input):
        # reshape the input from (batch_size x seq_len x input_size) to (batch_size x input_size x seq_len)
        # since that is how CNN expects it
        input = torch.transpose(input, 1, 2).unsqueeze_(3)

        input = self.conv(input)
        # the output of conv1d is expected to be (batch_size x output_channels(number of kernels) x seq_len)
        # but seq_len can be shorter, since it's valid cross-correlation not full cross-correlation

        # transpose input again since LSTM expects it as (batch_size x seq_len x input_size)
        # noinspection PyUnresolvedReferences

        # input = torch.transpose(input, 1, 2)
        input = input.view(-1, input.size()[1] * input.size()[2] * input.size()[3])

        # self.lstm.flatten_parameters()
        # output, (hidden, cell) = self.lstm(input, None)

        # the output is returned as (batch_number x sequence_length x hidden_size) since batch_first is set to true
        # otherwise, if the default settings are used, batch number comes second in dimensions
        # we are interested in only the output of the last time step, since this is a many to one architectures

        output = self.hidden(input)

        output = self.hidden2out(output)
        output = self.softmax(output)

        return output

    @property
    def name(self):
        return self.__name
