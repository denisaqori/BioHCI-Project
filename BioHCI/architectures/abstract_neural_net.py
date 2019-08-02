"""
Created: 8/2/19
© Denisa Qori McDonald 2019 All Rights Reserved
"""
from abc import abstractmethod

import torch.nn as nn

from BioHCI.definitions.neural_net_def import NeuralNetworkDefinition


class AbstractNeuralNetwork(nn.Module):
    def __init__(self):
        super(AbstractNeuralNetwork, self).__init__()

    @property
    @abstractmethod
    def name(self):
        pass
