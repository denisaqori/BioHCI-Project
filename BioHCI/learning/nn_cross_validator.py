import time
from typing import List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import BioHCI.helpers.type_aliases as types
from BioHCI.architectures.abstract_neural_net import AbstractNeuralNetwork
from BioHCI.data.data_splitter import DataSplitter
from BioHCI.data_processing.category_balancer import CategoryBalancer
from BioHCI.data_processing.feature_constructor import FeatureConstructor
from BioHCI.definitions.neural_net_def import NeuralNetworkDefinition
from BioHCI.definitions.study_parameters import StudyParameters
from BioHCI.helpers import utilities as utils
from BioHCI.knitted_components.knitted_component import KnittedComponent
from BioHCI.learning.cross_validator import CrossValidator
from BioHCI.learning.evaluator import Evaluator
from BioHCI.learning.trainer import Trainer


class NNCrossValidator(CrossValidator):

    def __init__(self, subject_dict: types.subj_dataset, data_splitter: DataSplitter, feature_constructor:
    FeatureConstructor, category_balancer: CategoryBalancer, neural_net: AbstractNeuralNetwork, parameters:
    StudyParameters, learning_def: NeuralNetworkDefinition, all_categories: List[str], knitted_component:
    KnittedComponent, extra_model_name: str = ""):
        assert (parameters.neural_net is True), "In StudyParameters, neural_net is set to False and you are " \
                                                "trying to instantiate a NNCrossValidator object!"
        # this list contains lists of accuracies for each epoch. There will be self._num_folds lists of _num_epochs
        # elements in this list after all training is done
        self.__all_epoch_train_accuracies = []
        self.__all_epoch_train_losses = []
        self.__all_epoch_val_accuracies = []
        self.__all_epoch_val_losses = []

        super(NNCrossValidator, self).__init__(subject_dict, data_splitter, feature_constructor, category_balancer,
                                               neural_net, parameters, learning_def, all_categories,
                                               knitted_component, extra_model_name)
        # the stochastic gradient descent function to update weights a self.perform_cross_validation()nd biases
        self.__optimizer = torch.optim.Adam(self.neural_net.parameters(), lr=learning_def.learning_rate)

        if parameters.classification:
            # the negative log likelihood loss function - useful to train classification problems with C classes
            self.__criterion = nn.NLLLoss()
        else:
            self.__criterion = nn.SmoothL1Loss()

    @property
    def all_epoch_train_accuracies(self) -> List[List[float]]:
        return self.__all_epoch_train_accuracies

    @property
    def all_epoch_train_losses(self) -> List[List[float]]:
        return self.__all_epoch_train_losses

    @property
    def all_epoch_val_accuracies(self) -> List[List[float]]:
        return self.__all_epoch_val_accuracies

    @property
    def all_epoch_val_losses(self) -> List[List[float]]:
        return self.__all_epoch_val_losses

    @property
    def optimizer(self):
        return self.__optimizer

    @property
    def criterion(self):
        return self.__criterion

    # implement the abstract method from the parent class CrossValidator; returns a dataset with labels wrapped in
    # the PyTorch DataLoader format
    def _get_data_and_labels(self, subj_dataset):
        if self.neural_net.name == "MLP":
            data, cat = self.get_all_subj_data(subj_dataset, seq=False)
        else:
            data, cat = self.get_all_subj_data(subj_dataset)

        # convert numpy ndarray to PyTorch tensor
        np_data = np.asarray(data, dtype=np.float32)
        data = torch.from_numpy(np_data)

        # convert categories from string to integer
        labels = utils.convert_categories(self.category_map, cat)

        # TODO: convert int_cat to yarn_positions by calling a function/property of touchpad
        if not self.parameters.classification:
            labels = self.knitted_component.get_button_centers(labels)

        labels = torch.from_numpy(labels)
        # the tensor_dataset is a tuple of TensorDataset type, containing a tensor with data (train or val),
        # and one with labels (train or val respectively)

        standardized_data = self.standardize(data)
        tensor_dataset = TensorDataset(standardized_data, labels)

        data_loader = DataLoader(tensor_dataset, batch_size=self.learning_def.batch_size,
                                 num_workers=self.parameters.num_threads, shuffle=True, pin_memory=True)

        return data_loader

    # implement the abstract method from the parent class CrossValidator; it is called for each fold in
    # cross-validation and after it trains for that fold, it appends the calculated losses and accuracies for each
    # epoch to the respective list in the CrossValidator object standout
    def train(self, train_dataset):
        train_data_loader = self._get_data_and_labels(train_dataset)
        trainer = Trainer(train_data_loader, self.neural_net, self.optimizer, self.criterion, self.knitted_component,
                          self.learning_def, self.parameters, self.writer, self.model_path)

        return trainer.loss, trainer.accuracy

    # evaluate the learning created during training on the validation dataset
    def val(self, val_dataset):
        val_data_loader = self._get_data_and_labels(val_dataset)
        model_to_eval = torch.load(self.model_path)

        evaluator = Evaluator(val_data_loader, model_to_eval, self.criterion, self.knitted_component,
                              self.confusion_matrix, self.learning_def, self.parameters, self.writer)

        fold_accuracy = evaluator.accuracy
        self.all_val_accuracies.append(fold_accuracy)

        return evaluator.loss, evaluator.accuracy

    def _specific_train_val(self, balanced_train, balanced_val):
        epoch_train_losses = []
        epoch_val_losses = []

        epoch_train_accuracies = []
        epoch_val_accuracies = []

        train_time_s = 0
        val_time_s = 0
        for epoch in range(1, self.learning_def.num_epochs + 1):

            train_start = time.time()
            current_train_loss, current_train_accuracy = self.train(balanced_train)
            train_time_diff = utils.time_diff(train_start)
            train_time_s += train_time_diff

            self.writer.add_scalar('Train Loss', current_train_loss, epoch)
            self.writer.add_scalar('Train Accuracy', current_train_accuracy, epoch)

            torch.save(self.neural_net, self.model_path)

            # start validating the learning
            val_start = time.time()
            current_val_loss, current_val_accuracy = self.val(balanced_val)
            val_time_diff = utils.time_diff(val_start)
            val_time_s += val_time_diff

            self.writer.add_scalar('Val Loss', current_val_loss, epoch)
            self.writer.add_scalar('Val Accuracy', current_val_accuracy, epoch)

            # Print epoch number, loss, accuracy, name and guess
            print_every = 10
            if epoch % print_every == 0:
                print(
                    f"Epoch {epoch}: Train Loss: {(current_train_loss / epoch):.5f}    Train Accuracy:"
                    f" {current_train_accuracy:.3f}    Val Loss: {(current_val_loss / epoch):.5f}    "
                    f"Val Accuracy: {current_val_accuracy:.3f}")

            # Add current loss avg to list of losses
            epoch_train_losses.append(current_train_loss / epoch)
            epoch_val_losses.append(current_val_loss / epoch)
            epoch_train_accuracies.append(current_train_accuracy)
            epoch_val_accuracies.append(current_val_accuracy)

            self.writer.add_scalar('Train Avg Loss', current_train_loss / epoch, epoch)
            self.writer.add_scalar('Val Avg Loss', current_val_loss / epoch, epoch)

        self.__all_epoch_train_accuracies.append(epoch_train_accuracies)
        self.__all_epoch_train_losses.append(epoch_train_losses)
        self.__all_epoch_val_accuracies.append(epoch_val_accuracies)
        self.__all_epoch_val_losses.append(epoch_val_losses)

        self.train_time = utils.time_s_to_str(train_time_s)
        self.val_time = utils.time_s_to_str(val_time_s)

        # torch.save(self.neural_net, self.model_path)

    def _store_specific_results(self):
        # accuracies for each epoch and each fold are added to the list that belongs only to this class
        # "_all_epoch_train_accuracies". The last accuracy of each train epoch is added to the list
        # "_all_train_accuracies, belonging more generally to the parent class
        for i in range(0, self.num_folds):
            self.all_train_accuracies.append(self.all_epoch_train_accuracies[i][-1])
        # self.all_epoch_train_accuracies.append(self.all_epoch_train_accuracies)
        print("Train Epoch Accuracies: ", self.all_epoch_train_accuracies)

        for i in range(0, self.num_folds):
            self.all_val_accuracies.append(self.all_epoch_val_accuracies[i][-1])
        # self.all_epoch_val_accuracies.append(self.all_epoch_val_accuracies)
        print("Val Epoch Accuracies: ", self.all_epoch_val_accuracies)

    def _log_specific_results(self):
        self.result_logger.debug(f"All fold train accuracies (all epochs): ")
        log_every = 10
        for i, ls in enumerate(self.all_epoch_train_accuracies):
            if log_every % 10 == 0:
                self.result_logger.debug(self._format_list(ls))

        self.result_logger.debug(f"All fold train losses (all epochs): ")
        log_every = 10
        for i, ls in enumerate(self.all_epoch_train_losses):
            if log_every % 10 == 0:
                self.result_logger.debug(self._format_list(ls))

        # self.result_logger.info(f"All fold train accuracies: {self._format_list(self.all_train_accuracies)}")
        # self.result_logger.info(f"All fold val accuracies: {self._format_list(self.all_val_accuracies)}")
        self.result_logger.info(f"Average train accuracy: {self.avg_train_accuracy:.2f}")

        # self.result_logger.info(f"All fold validation accuracies: {self._format_list(self.all_val_accuracies)}")
        # self.result_logger.info(f"All fold validation accuracies: {self._format_list(self.all_val_accuracies)}")
        self.result_logger.debug(f"All fold val accuracies (all epochs): ")
        log_every = 10
        for i, ls in enumerate(self.all_epoch_val_accuracies):
            if log_every % 10 == 0:
                self.result_logger.debug(self._format_list(ls))

        self.result_logger.debug(f"All fold val losses (all epochs): ")
        log_every = 10
        for i, ls in enumerate(self.all_epoch_val_losses):
            if log_every % 10 == 0:
                self.result_logger.debug(self._format_list(ls))

        self.result_logger.info(f"Average validation accuracy: {self.avg_val_accuracy:.2f}\n")

    def _format_list(self, float_ls: List[float]) -> List[str]:
        my_formatted_list = [   '%.2f' % elem for elem in float_ls]
        return my_formatted_list
