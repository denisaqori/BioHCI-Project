from BioHCI.architectures.abstract_neural_net import AbstractNeuralNetwork
import torch
import torch.nn as nn
from torch.autograd import Variable


class EncoderLSTM(nn.Module):
    def __init__(self, nn_learning_def):
        super(EncoderLSTM, self).__init__()

        self.input_size = nn_learning_def.input_size
        self.hidden_size = nn_learning_def.num_hidden
        self.num_layers = nn_learning_def.num_layers

        self.lstm = nn.LSTM(input_size=self.hidden_size, hidden_size=self.hidden_size, num_layers=self.num_layers)

    def forward(self, inputs, hidden):
        # Note: we run this all at once (over the whole input sequence)
        output, hidden = self.lstm(inputs, hidden)
        return output, hidden

    def init_hidden(self):
        if self.use_cuda:
            return (Variable(torch.zeros(1, self.batch_size, self.hidden_size)).float().cuda(),
                    Variable(torch.zeros(1, self.batch_size, self.hidden_size)).float().cuda())
        else:
            return (Variable(torch.zeros(1, self.batch_size, self.hidden_size)).float(),
                    Variable(torch.zeros(1, self.batch_size, self.hidden_size)).float())

        # hidden = Variable(torch.zeros(self.n_layers, 1, self.hidden_size)).cuda()


class AttnDecoderLSTM(AbstractNeuralNetwork):

    def __init__(self, nn_learning_def):
        super(AttnDecoderLSTM, self).__init__()

        self.__name = "AttnDecoderLSTM"
        assert self.__name == nn_learning_def.nn_name

        self.input_size = nn_learning_def.input_size
        self.hidden_size = nn_learning_def.num_hidden
        self.output_size = nn_learning_def.output_size
        self.batch_size = nn_learning_def.batch_size
        self.batch_first = nn_learning_def.batch_first
        self.dropout_rate = nn_learning_def.dropout_rate
        self.num_layers = nn_learning_def.num_layers
        self.use_cuda = nn_learning_def.use_cuda

        # defining layers
        self.attn = nn.Linear(self.hidden_size, self.hidden_size)
        self.lstm = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=self.num_layers,
                            dropout=self.dropout_rate, batch_first=self.batch_first, bidirectional=False)

        # a linear layer mapping the hidden state to output, then squashing
        # the output (probability for each class) through a softmax function
        self.hidden2out = nn.Linear(self.hidden_size, self.output_size)
        # define the softmax function, declaring the dimension along which it will be computed (so every slice along it
        # will sum to 1). The output  (on which the function will be called) will have the shape batch_size x
        # output_size
        self.softmax = nn.LogSoftmax(dim=1)  # already ensured 1 is the right dimension and calculation is correct

    # The hidden and cell state dimensions are: (num_layers * num-direction, batch, hidden_size) for each
    def init_hidden(self):
        if self.use_cuda:
            return (Variable(torch.zeros(1, self.batch_size, self.hidden_size)).float().cuda(),
                    Variable(torch.zeros(1, self.batch_size, self.hidden_size)).float().cuda())
        else:
            return (Variable(torch.zeros(1, self.batch_size, self.hidden_size)).float(),
                    Variable(torch.zeros(1, self.batch_size, self.hidden_size)).float())

    def forward(self, input, last_hidden, encoder_outputs):
        # (h_0, c_0) assumed zero even if given (but we have defined them above)
        # this is the main layer of the architectures where all the gradient computation happens
        # the whole sequence is passed through there
        # output contains result for each time step, hidden only for last, and cell state contains the
        # cell state of the last time step

        # Calculate attention weights and apply to encoder outputs
        attn_weights = self.attn(last_hidden[-1], encoder_outputs)


        self.lstm.flatten_parameters()
        output, (hidden, cell) = self.lstm(input, None)

        # the output is returned as (batch_number x sequence_length x hidden_size) since batch_first is set to true
        # otherwise, if the default settings are used, batch number comes second in dimensions
        # we are interested in only the output of the last time step, since this is a many to one architectures
        if self.batch_first:
            output = self.hidden2out(output[:, -1, :])
        else:
            output = self.hidden2out(output[-1, :, :])

        output = self.softmax(output)
        return output

    @property
    def name(self) -> str:
        return self.__name
