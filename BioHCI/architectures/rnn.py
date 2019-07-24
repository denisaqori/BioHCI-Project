import torch.nn as nn
from torch.autograd import Variable
import torch


# Class based on PyTorch sample code from Sean Robertson (Classifying Names with a Character-Level RNN)
# http://pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial.html 

# !!!!!!!!!!! DOES NOT WORK AT THE MOMENT!!!!!!!!!!!!!!!!!!!
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()

        self.name = "RNN"
        self.hidden_size = hidden_size

        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.softmax = nn.LogSoftmax()

    # self.relu = nn.ReLU()

    def forward(self, input, hidden):
        # print ("Input dim:", input.size())
        # print ("Hidden dim:", hidden.size())
        combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        # print("Size of output returned by the rnn layer: ", output.size())
        # print("Size of hidden[0] returned by the rnn layer: ", hidden.size())
        # print("Size of hidden[1] returned by the lstm layer: ", hidden[1].size())
        output = self.softmax(output)
        print("Size of output returned by the softmax function: ", output.size())
        return output, hidden

    def init_hidden(self) -> object:
        # The hidden and cell dimensions are: (num_layers, batch, hidden_size)
        return Variable(torch.zeros(1, self.hidden_size))

    def name(self):
        return "RNN"
