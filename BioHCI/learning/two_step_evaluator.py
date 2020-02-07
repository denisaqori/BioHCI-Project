"""
Created: 1/7/20
© Denisa Qori McDonald 2020 All Rights Reserved
"""
from typing import List

import numpy as np
import torch
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader

from BioHCI.architectures.abstract_neural_net import AbstractNeuralNetwork
from BioHCI.definitions.learning_def import LearningDefinition
from BioHCI.definitions.study_parameters import StudyParameters
from BioHCI.knitted_components.knitted_component import KnittedComponent
from BioHCI.learning.evaluator import Evaluator


# TODO: redo trainer and evaluator to inherit from the same parent class - too much copy-paste
class TwoStepEvaluator(Evaluator):
    def __init__(self, row_data_loader: DataLoader, column_data_loader: DataLoader,
                 row_model: AbstractNeuralNetwork, column_model: AbstractNeuralNetwork,
                 criterion, knitted_component: KnittedComponent, row_confusion: np.ndarray, column_confusion:
                 np.ndarray, button_confusion: np.ndarray, neural_network_def: LearningDefinition, parameters:
                 StudyParameters, summary_writer: SummaryWriter):
        self.__column_data_loader = column_data_loader
        self.__knitted_component = knitted_component
        self.__column_model = column_model
        super(TwoStepEvaluator, self).__init__(row_data_loader, row_model, criterion, row_confusion,
                                               neural_network_def, parameters, summary_writer)

        self.row_loss, self.row_accuracy, self.column_loss, self.column_accuracy, self.button_accuracy = \
            self.evaluate (row_data_loader, row_confusion, column_confusion, button_confusion)

    @property
    def column_data_loader(self) -> DataLoader:
        return self.__column_data_loader

    @property
    def column_model(self) -> AbstractNeuralNetwork:
        return self.__column_model

    def evaluate(self, row_data_loader: DataLoader, row_confusion: np.ndarray,
                 column_confusion: np.ndarray = None, button_confusion: np.ndarray = None):
        row_correct = 0
        column_correct = 0
        button_correct = 0

        row_total = 0
        column_total = 0
        button_total = 0

        row_loss = 0
        column_loss = 0

        column_it = iter(self.column_data_loader)
        # Go through the test dataset and record which are correctly guessed
        for step, (row_data_chunk, row_category) in enumerate(row_data_loader):
            # get the same batch of data from the column data loader with the corresponding labels
            column_data_chunk, column_category = next(column_it)
            assert torch.eq(row_data_chunk, column_data_chunk).all()

            data_chunk = row_data_chunk.float()
            # data_chunk_tensor has shape (batch_size x samples_per_step x num_attr)
            # category_tensor has shape (batch_size)
            # batch_size is passed as an argument to train_data_loader
            if self._parameters.classification:
                row_category = row_category.long()  # the loss function requires it
                column_category = column_category.long()  # the loss function requires it
            else:
                row_category = row_category.float()
                column_category = column_category.float()

            # get row category
            row_output, row_loss = self.evaluate_chunks_in_batch(data_chunk, row_category, self._model_to_eval)
            row_loss += row_loss

            # get column prediction
            column_output, column_loss = self.evaluate_chunks_in_batch(data_chunk, column_category, self.column_model)
            column_loss += column_loss

            # for every element of the batch
            for i in range(0, len(row_category)):
                assert self._parameters.classification

                row_total += 1
                row_category_i = int(row_category[i])
                row_predicted_i = self._category_from_output(row_output[i])

                # adding data to the matrix
                row_confusion[row_category_i][row_predicted_i] += 1
                if row_category_i == row_predicted_i:
                    row_correct += 1

                column_total += 1
                column_category_i = int(column_category[i])
                column_predicted_i = self._category_from_output(column_output[i])

                column_confusion[column_category_i][column_predicted_i] += 1
                if column_category_i == column_predicted_i:
                    column_correct += 1

                # calculate predicted and real button based on corresponding rows and columns
                button_total += 1
                button_predicted_i = self.__knitted_component.get_button_position(row_predicted_i, column_predicted_i)
                button_category_i = self.__knitted_component.get_button_position(row_category_i, column_category_i)

                button_confusion[button_category_i][button_predicted_i] += 1
                if button_predicted_i == button_category_i:
                    button_correct += 1

        row_accuracy = row_correct / row_total
        column_accuracy = column_correct / column_total
        button_accuracy = button_correct / button_total

        return row_loss, row_accuracy, column_loss, column_accuracy, button_accuracy
