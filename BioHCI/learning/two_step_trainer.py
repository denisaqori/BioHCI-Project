"""
Created: 1/7/20
© Denisa Qori McDonald 2020 All Rights Reserved
"""
from typing import List

import torch
from tensorboardX import SummaryWriter
from torch.autograd import Variable
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader

from BioHCI.architectures.abstract_neural_net import AbstractNeuralNetwork
from BioHCI.definitions.neural_net_def import NeuralNetworkDefinition
from BioHCI.definitions.study_parameters import StudyParameters
from BioHCI.learning.trainer import Trainer


class TwoStepTrainer(Trainer):
    def __init__(self, train_data_loader: DataLoader, secondary_data_loader: DataLoader, neural_net:
    AbstractNeuralNetwork, secondary_neural_net_ls: List[AbstractNeuralNetwork], optimizer: Optimizer, criterion,
    knitted_component, neural_network_def: NeuralNetworkDefinition, secondary_learning_def: NeuralNetworkDefinition,
    parameters: StudyParameters, summary_writer: SummaryWriter, model_path: str) -> None:

        self.__button_data_loader = secondary_data_loader
        self.__button_nn_ls = secondary_neural_net_ls
        self.__button_learning_def = secondary_learning_def
        self.__knitted_component = knitted_component
        self.secondary_data_loader = secondary_data_loader

        super(TwoStepTrainer, self).__init__(train_data_loader, neural_net, optimizer, criterion, neural_network_def,
                                             parameters, summary_writer, model_path)
    @@property
    def button_nn_ls(self) -> List[AbstractNeuralNetwork]:
        return self.__button_nn_ls

    def __category_from_output(self, output):
        return self.__knitted_component.get_button_id(output)

    #TODO: Add proper optimizers in addition to neural networks!!!!!!!!!!!!!!!!
    def _train(self, primary_data_loader):
        row_correct = 0
        button_correct = 0

        row_total = 0
        button_total = 0

        row_loss = 0
        button_loss = 0

        button_it = iter(self.secondary_data_loader)
        # goes through the whole training dataset in tensor chunks and batches computing output and loss
        for step, (row_data_chunk, row_category) in enumerate(primary_data_loader):  # gives batch data
            # get the same batch of data from the secondary data loader with the corresponding labels
            button_data_chunk, button_category = next(button_it)
            assert torch.eq(row_data_chunk, button_data_chunk).all()

            data_chunk = row_data_chunk.float()
            if step == 0:
                input = Variable(data_chunk)
                self._writer.add_graph(self._neural_net, input.cuda(), True)

            # data_chunk_tensor has shape (batch_size x samples_per_chunk x num_attr)
            # category_tensor has shape (batch_size)
            # batch_size is passed as an argument to train_data_loader
            if self._parameters.classification:
                row_category = row_category.long()  # the loss function requires it
            else:
                row_category = row_category.float()

            row_output, row_loss = self._train_chunks_in_batch(row_category, data_chunk, self._neural_net)
            row_loss += row_loss

            # for every element of the batch calculate how many times the row was predicted correctly
            for i in range(0, len(row_category)):
                row_total = row_total + 1

                # calculating predicted categories for the whole batch
                assert self._parameters.classification
                row_predicted_i = self.__category_from_output(row_output[i])

                row_category_i = int(row_category[i])
                if row_category_i == row_predicted_i:
                    row_correct += 1

            button_nn = self.button_nn_ls[]

            # button_output, button_loss = self.______________________train_chunks_in_batch()


        # for name, param in self.__neural_net.named_parameters():
        #     self.__writer.add_histogram(name, param.clone().cpu().data.numpy(), epoch)

        row_accuracy = row_correct / row_total

        return row_loss, row_accuracy

