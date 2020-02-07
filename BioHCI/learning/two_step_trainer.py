"""
Created: 1/7/20
© Denisa Qori McDonald 2020 All Rights Reserved
"""
from typing import List

import torch
from tensorboardX import SummaryWriter
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader

from BioHCI.architectures.abstract_neural_net import AbstractNeuralNetwork
from BioHCI.definitions.neural_net_def import NeuralNetworkDefinition
from BioHCI.definitions.study_parameters import StudyParameters
from BioHCI.learning.trainer import Trainer


class TwoStepTrainer(Trainer):
    def __init__(self, row_data_loader: DataLoader, column_data_loader: DataLoader, row_neural_net:
    AbstractNeuralNetwork, column_neural_net: AbstractNeuralNetwork, row_optimizer: Optimizer,
                 column_optimizer: Optimizer, criterion, knitted_component, row_learning_def:
            NeuralNetworkDefinition, column_learning_def: NeuralNetworkDefinition, parameters: StudyParameters,
                 summary_writer: SummaryWriter, row_model_path: str) -> None:

        self.__knitted_component = knitted_component
        self.__row_nn = row_neural_net
        self.__column_nn = column_neural_net
        self.__column_learning_def = column_learning_def
        self.__column_optimizer = column_optimizer

        super(TwoStepTrainer, self).__init__(row_data_loader, row_neural_net, row_optimizer, criterion,
                                             row_learning_def, parameters, summary_writer, row_model_path)

        self.__row_loss, self.__row_accuracy, self.__column_loss, self.__column_accuracy, self.__button_accuracy = \
            self._train(row_data_loader, column_data_loader)

    @property
    def row_nn(self):
        return self.__row_nn

    @property
    def row_optim(self):
        return self._optimizer

    @property
    def column_nn(self) -> AbstractNeuralNetwork:
        return self.__column_nn

    @property
    def column_optim(self) -> Optimizer:
        return self.__column_optimizer

    # def __category_from_output(self, output):
    #     return self.__knitted_component.get_button_id(output)

    def _train(self, *data_loaders):
        row_data_loader = data_loaders[0]
        column_data_loader = data_loaders[1]

        row_correct = 0
        column_correct = 0
        button_correct = 0

        row_total = 0
        column_total = 0
        button_total = 0

        row_loss = 0
        column_loss = 0

        column_it = iter(column_data_loader)
        # goes through the whole training dataset in tensor chunks and batches computing output and loss
        for step, (row_data_chunk, row_category) in enumerate(row_data_loader):  # gives batch data
            # get the same batch of data from the column data loader with the corresponding labels
            column_data_chunk, column_category = next(column_it)
            assert torch.eq(row_data_chunk, column_data_chunk).all()

            data_chunk = row_data_chunk.float()

            # data_chunk_tensor has shape (batch_size x samples_per_chunk x num_attr)
            # category_tensor has shape (batch_size)
            # batch_size is passed as an argument to train_data_loader
            if self._parameters.classification:
                row_category = row_category.long()  # the loss function requires it
                column_category = column_category.long()  # the loss function requires it
            else:
                row_category = row_category.float()
                column_category = column_category.float()

            # get row prediction
            row_output, row_loss = self._train_chunks_in_batch(row_category, data_chunk, self.row_nn, self.row_optim)
            row_loss += row_loss

            # get column prediction
            column_output, column_loss = self._train_chunks_in_batch(column_category, data_chunk, self.column_nn,
                                                                    self.column_optim)
            column_loss += column_loss

            # for every element of the batch calculate how many times the row, column and button were predicted
            # correctly
            for i in range(0, len(row_category)):
                row_total += 1
                assert self._parameters.classification

                # calculating predicted categories for the whole batch
                row_predicted_i = self._category_from_output(row_output[i])
                row_category_i = int(row_category[i])
                if row_category_i == row_predicted_i:
                    row_correct += 1

                column_total += 1
                column_predicted_i = self._category_from_output(column_output[i])
                column_category_i = int(column_category[i])
                if column_category_i == column_predicted_i:
                    column_correct += 1

                # calculate predicted and real button based on corresponding rows and columns
                button_predicted_i = self.__knitted_component.get_button_position(row_predicted_i, column_predicted_i)
                button_category_i = self.__knitted_component.get_button_position(row_category_i, column_category_i)

                button_total += 1
                if button_predicted_i == button_category_i:
                    button_correct += 1

        row_accuracy = row_correct / row_total
        column_accuracy = column_correct / column_total
        button_accuracy = button_correct / button_total

        return row_loss, row_accuracy, column_loss, column_accuracy, button_accuracy

    @property
    def row_loss(self):
        return self.__row_loss

    @property
    def row_accuracy(self):
        return self.__row_accuracy

    @property
    def column_loss(self):
        return self.__column_loss

    @property
    def column_accuracy(self):
        return self.__column_accuracy

    @property
    def button_accuracy(self):
        return self.__button_accuracy
