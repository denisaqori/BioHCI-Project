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
    def __init__(self, train_data_loader: DataLoader, secondary_data_loader: DataLoader, neural_net:
    AbstractNeuralNetwork, secondary_neural_net_ls: List[AbstractNeuralNetwork], optimizer: Optimizer,
                 secondary_optimizer_ls: List[Optimizer], criterion, knitted_component, neural_network_def:
            NeuralNetworkDefinition, secondary_learning_def: NeuralNetworkDefinition, parameters: StudyParameters,
                 summary_writer: SummaryWriter, model_path: str) -> None:

        self.__button_nn_ls = secondary_neural_net_ls
        self.__button_learning_def = secondary_learning_def
        self.__knitted_component = knitted_component
        self.secondary_data_loader = secondary_data_loader
        self.__secondary_optimizer_ls = secondary_optimizer_ls

        super(TwoStepTrainer, self).__init__(train_data_loader, neural_net, optimizer, criterion, neural_network_def,
                                             parameters, summary_writer, model_path)
        self.__row_loss, self.__row_accuracy, self.__button_loss, self.__button_accuracy = self._train(
            train_data_loader)

    @property
    def button_nn_ls(self) -> List[AbstractNeuralNetwork]:
        return self.__button_nn_ls

    @property
    def button_optim_ls(self) -> List[Optimizer]:
        return self.__secondary_optimizer_ls

    # def __category_from_output(self, output):
    #     return self.__knitted_component.get_button_id(output)

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
            # if step == 0:
            #     input = Variable(data_chunk)
            #     self._writer.add_graph(self._neural_net, input.cuda(), True)

            # data_chunk_tensor has shape (batch_size x samples_per_chunk x num_attr)
            # category_tensor has shape (batch_size)
            # batch_size is passed as an argument to train_data_loader
            if self._parameters.classification:
                row_category = row_category.long()  # the loss function requires it
                button_category = button_category.long()  # the loss function requires it
            else:
                row_category = row_category.float()
                button_category = button_category.float()

            row_output, row_loss = self._train_chunks_in_batch(row_category, data_chunk, self._neural_net,
                                                               self._optimizer)
            row_loss += row_loss

            # for every element of the batch calculate how many times the row was predicted correctly
            for i in range(0, len(row_category)):
                row_total = row_total + 1

                # calculating predicted categories for the whole batch
                assert self._parameters.classification
                row_predicted_i = self._category_from_output(row_output[i])

                row_category_i = int(row_category[i])
                if row_category_i == row_predicted_i:
                    row_correct += 1

                # get the corresponding neural network and optimizer for the correct row
                button_nn = self.button_nn_ls[row_category_i]
                button_optim = self.button_optim_ls[row_category_i]

                # to input into the secondary network - convert real label to one (0, num_columns)
                button_cat = (button_category[i] % self.__knitted_component.num_cols).unsqueeze(0)
                button_data = button_data_chunk[i].unsqueeze(0)
                # train using that one instance
                button_output, button_loss = self._train_chunks_in_batch(button_cat, button_data, button_nn,
                                                                         button_optim)
                button_loss += button_loss
                button_predicted_i = self._category_from_output(button_output)
                # convert back to real label
                button_predicted_i = button_predicted_i + row_category_i * self.__knitted_component.num_cols

                button_total += 1
                if button_predicted_i == int(button_category[i]):
                    button_correct += 1

        # for name, param in self.__neural_net.named_parameters():
        #     self.__writer.add_histogram(name, param.clone().cpu().data.numpy(), epoch)

        row_accuracy = row_correct / row_total
        button_accuracy = button_correct / button_total

        return row_loss, row_accuracy, button_loss, button_accuracy

    @property
    def row_loss(self):
        return self.__row_loss

    @property
    def row_accuracy(self):
        return self.__row_accuracy

    @property
    def button_loss(self):
        return self.__button_loss

    @property
    def button_accuracy(self):
        return self.__button_accuracy
