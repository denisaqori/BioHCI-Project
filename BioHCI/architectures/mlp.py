"""
Created: 10/2/19
© Denisa Qori McDonald 2019 All Rights Reserved
"""
import torch.nn as nn
from BioHCI.architectures.abstract_neural_net import AbstractNeuralNetwork


class MLP(AbstractNeuralNetwork):

    def __init__(self, nn_learning_def):
        super(MLP, self).__init__()

        self.__name = "MLP"
        self.input_size = nn_learning_def.input_size
        self.hidden_size = nn_learning_def.num_hidden
        self.output_size = nn_learning_def.output_size
        self.batch_size = nn_learning_def.batch_size
        self.batch_first = nn_learning_def.batch_first
        self.use_cuda = nn_learning_def.use_cuda

        self.mlp = nn.Sequential(
            nn.Linear(self.input_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.output_size)
            )

        self.layer1 = nn.Linear(self.input_size, self.hidden_size)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(self.hidden_size, self.output_size)

        self.softmax = nn.LogSoftmax(dim=1)  # already ensured 1 is the right dimension and calculation is correct
        # self.relu = nn.ReLU(dim=1)

    def forward(self, input):
        # input = input.view(-1, self.num_flat_features(input))

        output = self.mlp(input)
        output = self.softmax(output)
        return output

    def num_flat_features(self, x):
        if self.batch_first:
            size = x.size()[1:]  # all dimensions except the batch dimension
        else:
            size = x.size()
            del size[0]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

    @property
    def name(self) -> str:
        return self.__name
