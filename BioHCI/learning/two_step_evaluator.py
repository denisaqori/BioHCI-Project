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
    def __init__(self, val_data_loader: DataLoader, secondary_data_loader: DataLoader,
                 model_to_eval: AbstractNeuralNetwork, secondary_models_ls: List[AbstractNeuralNetwork],
                 criterion, knitted_component: KnittedComponent, confusion: np.ndarray, secondary_confusion_ls: List[
                 np.ndarray], neural_network_def: LearningDefinition, parameters: StudyParameters,
                 summary_writer: SummaryWriter):
        self.__secondary_data_loader = secondary_data_loader
        self.__knitted_component = knitted_component
        self.__button_model_ls = secondary_models_ls
        super(TwoStepEvaluator, self).__init__(val_data_loader, model_to_eval, criterion, confusion,
                                               neural_network_def, parameters, summary_writer)

        self.row_loss, self.row_accuracy, self.button_loss, self.button_accuracy = self.evaluate(
            self.__val_data_loader, confusion, secondary_confusion_ls)

    @property
    def secondary_data_loader(self) -> DataLoader:
        return self.__secondary_data_loader

    @property
    def button_model_ls(self) -> List[AbstractNeuralNetwork]:
        return self.__button_model_ls

    def evaluate(self, primary_data_loader: DataLoader, confusion_primary: np.ndarray,
                 confusion_secondary: np.ndarray = None):
        row_correct = 0
        button_correct = 0

        row_total = 0
        button_total = 0

        row_loss = 0
        button_loss = 0

        button_it = iter(self.secondary_data_loader)
        # Go through the test dataset and record which are correctly guessed
        for step, (row_data_chunk, row_category) in enumerate(primary_data_loader):
            # get the same batch of data from the secondary data loader with the corresponding labels
            button_data_chunk, button_category = next(button_it)
            assert torch.eq(row_data_chunk, button_data_chunk).all()

            data_chunk = row_data_chunk.float()
            # data_chunk_tensor has shape (batch_size x samples_per_step x num_attr)
            # category_tensor has shape (batch_size)
            # batch_size is passed as an argument to train_data_loader
            if self._parameters.classification:
                row_category = row_category.long()  # the loss function requires it
                button_category = button_category.long()  # the loss function requires it
            else:
                row_category = row_category.float()
                button_category = button_category.float()

            # getting the architectures guess for the category
            row_output, row_loss = self.evaluate_chunks_in_batch(data_chunk, row_category, self._model_to_eval)
            row_loss += row_loss

            # for every element of the batch
            for i in range(0, len(row_category)):
                row_total = row_total + 1
                # calculating true category
                row_category_i = int(row_category[i])

                # calculating predicted categories for the whole batch
                assert self._parameters.classification
                row_predicted_i = self._category_from_output(row_output[i])

                # adding data to the matrix
                confusion_primary[row_category_i][row_predicted_i] += 1

                if row_category_i == row_predicted_i:
                    row_correct += 1

                # get the corresponding neural network and optimizer for the correct row
                button_model = self.button_model_ls[row_category_i]

                # to input into the secondary network - convert real label to one (0, num_columns)
                button_cat = (button_category[i] % self.__knitted_component.num_cols).unsqueeze(0)
                button_data = button_data_chunk[i].unsqueeze(0)
                # train using that one instance
                button_output, button_loss = self.evaluate_chunks_in_batch(button_cat, button_data, button_model)
                button_loss += button_loss

                button_predicted_i = self._category_from_output(button_output)
                # convert back to real label
                button_predicted_i = button_predicted_i + row_category_i * self.__knitted_component.num_cols

                button_total += 1
                if button_predicted_i == int(button_category[i]):
                    button_correct += 1

                confusion_secondary[button_category[i][button_predicted_i]] += 1

        row_accuracy = row_correct / row_total
        button_accuracy = button_correct / button_total

        return row_loss, row_accuracy, button_loss, button_accuracy
