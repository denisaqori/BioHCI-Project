"""
Created: 1/7/20
© Denisa Qori McDonald 2020 All Rights Reserved
"""

from tensorboardX import SummaryWriter
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader

from BioHCI.architectures.abstract_neural_net import AbstractNeuralNetwork
from BioHCI.definitions.neural_net_def import NeuralNetworkDefinition
from BioHCI.definitions.study_parameters import StudyParameters
from BioHCI.learning.trainer import Trainer


class TwoStepTrainer(Trainer):
    def __init__(self, train_data_loader: DataLoader, secondary_data_loader: DataLoader, neural_net:
    AbstractNeuralNetwork, secondary_neural_net: AbstractNeuralNetwork, optimizer: Optimizer, criterion,
    knitted_component, neural_network_def: NeuralNetworkDefinition, secondary_learning_def: NeuralNetworkDefinition,
    parameters: StudyParameters, summary_writer: SummaryWriter, model_path: str) -> None:

        self.__button_data_loader = secondary_data_loader
        self.__button_nn = secondary_neural_net
        self.__button_learning_def = secondary_learning_def
        self.__knitted_component = knitted_component
        self.__nn_array = []

        super(TwoStepTrainer, self).__init__(train_data_loader, neural_net, optimizer, criterion, neural_network_def,
                                             parameters, summary_writer, model_path)

    def __category_from_output(self, output):
        return self.__knitted_component.get_button_id(output)

    def __populate_nn_array(self):
        for _ in self.__knitted_component.num_rows:
            self.__nn_array.append(self.__button_nn)
