"""
Created: 1/7/20
© Denisa Qori McDonald 2020 All Rights Reserved
"""
import time
from os.path import join
from typing import List

import numpy as np
import torch
from torch.optim import Optimizer
from torch.utils.data import DataLoader, TensorDataset

import BioHCI.helpers.type_aliases as types
import BioHCI.helpers.utilities as utils
from BioHCI.architectures.abstract_neural_net import AbstractNeuralNetwork
from BioHCI.data.data_splitter import DataSplitter
from BioHCI.data_processing.category_balancer import CategoryBalancer
from BioHCI.data_processing.feature_constructor import FeatureConstructor
from BioHCI.definitions.neural_net_def import NeuralNetworkDefinition
from BioHCI.definitions.study_parameters import StudyParameters
from BioHCI.knitted_components.knitted_component import KnittedComponent
from BioHCI.learning.nn_analyser import NNAnalyser
from BioHCI.learning.two_step_evaluator import TwoStepEvaluator
from BioHCI.learning.two_step_trainer import TwoStepTrainer


#TODO: fix the way column nn is created - a new one needs to be created for each cv fold
class KnittingCrossValidator(NNAnalyser):

    def __init__(self, subject_dict: types.subj_dataset, data_splitter: DataSplitter,
                 feature_constructor: FeatureConstructor, category_balancer: CategoryBalancer,
                 column_neural_net: AbstractNeuralNetwork, parameters: StudyParameters,
                 row_learning_def: NeuralNetworkDefinition, column_learning_def: NeuralNetworkDefinition,
                 all_categories: List[str], knitted_component: KnittedComponent, extra_model_name: str = ""):

        self.__knitted_component = knitted_component
        assert (parameters.neural_net is True), "In StudyParameters, neural_net is set to False and you are " \
                                                "trying to instantiate a NNCrossValidator object!"
        self.__column_nn = column_neural_net
        self.__column_learning_def = column_learning_def
        self.__column_confusion = np.zeros((knitted_component.num_cols, knitted_component.num_cols))
        self.__button_confusion = np.zeros((knitted_component.num_buttons, knitted_component.num_buttons))

        # TODO: add a list of models created and make sure that things are saved properly - might be challenging

        self.__column_optim = torch.optim.Adam(column_neural_net.parameters(), lr=self.column_learning_def.learning_rate)

        self.__all_epoch_train_row_accuracies = []
        self.__all_epoch_train_row_losses = []
        self.__all_epoch_train_col_accuracies = []
        self.__all_epoch_train_col_losses = []
        self.__all_epoch_train_button_accuracies = []
        self.__all_epoch_train_button_losses = []

        self.__all_epoch_val_row_accuracies = []
        self.__all_epoch_val_row_losses = []
        self.__all_epoch_val_col_accuracies = []
        self.__all_epoch_val_col_losses = []
        self.__all_epoch_val_button_accuracies = []
        self.__all_epoch_val_button_losses = []

        self.__row_fold_accuracies = []
        self.__col_fold_accuracies = []
        self.__button_fold_accuracies = []

        super(KnittingCrossValidator, self).__init__(subject_dict, data_splitter, feature_constructor,
                                                     category_balancer, parameters, row_learning_def,
                                                     all_categories, extra_model_name)
        row_model_name = self._produce_model_name(extra="_r")
        self.__row_model_path = join(self.model_dir, row_model_name)

        column_model_name = self._produce_model_name(extra="_c")
        self.__column_model_path = join(self.model_dir, column_model_name)

    @property
    def _all_epoch_train_row_accuracies(self) -> List[List[float]]:
        return self.__all_epoch_train_row_accuracies

    @property
    def _all_epoch_train_row_losses(self) -> List[List[float]]:
        return self.__all_epoch_train_row_losses

    @property
    def _all_epoch_train_col_accuracies(self) -> List[List[float]]:
        return self.__all_epoch_train_col_accuracies

    @property
    def _all_epoch_train_col_losses(self) -> List[List[float]]:
        return self.__all_epoch_train_col_losses

    @property
    def _all_epoch_train_button_accuracies(self) -> List[List[float]]:
        return self.__all_epoch_train_button_accuracies

    @property
    def _all_epoch_train_button_losses(self) -> List[List[float]]:
        return self.__all_epoch_train_button_losses

    @property
    def _all_epoch_val_row_accuracies(self) -> List[List[float]]:
        return self.__all_epoch_val_row_accuracies

    @property
    def _all_epoch_val_row_losses(self) -> List[List[float]]:
        return self.__all_epoch_val_row_losses

    @property
    def _all_epoch_val_col_accuracies(self) -> List[List[float]]:
        return self.__all_epoch_val_col_accuracies

    @property
    def _all_epoch_val_col_losses(self) -> List[List[float]]:
        return self.__all_epoch_val_col_losses

    @property
    def _all_epoch_val_button_accuracies(self) -> List[List[float]]:
        return self.__all_epoch_val_button_accuracies

    @property
    def _all_epoch_val_button_losses(self) -> List[List[float]]:
        return self.__all_epoch_val_button_losses

    @property
    def row_fold_accuracies(self) -> List[List[float]]:
        return self.__row_fold_accuracies

    @property
    def col_fold_accuracies(self) -> List[List[float]]:
        return self.__col_fold_accuracies

    @property
    def button_fold_accuracies(self) -> List[List[float]]:
        return self.__button_fold_accuracies

    @property
    def knitted_component(self) -> KnittedComponent:
        return self.__knitted_component

    @property
    def row_learning_def(self):
        return self.learning_def

    @property
    def column_learning_def(self) -> NeuralNetworkDefinition:
        return self.__column_learning_def

    @property
    def column_neural_net(self) -> AbstractNeuralNetwork:
        return self.__column_nn

    @property
    def row_model_path(self) -> str:
        return self.__row_model_path

    @row_model_path.setter
    def row_model_path(self, row_model_path):
        self.__row_model_path = row_model_path

    @property
    def column_model_path(self) -> str:
        return self.__column_model_path

    @column_model_path.setter
    def column_model_path(self, column_model_path):
        self.__column_model_path = column_model_path

    @property
    def column_optim(self) -> Optimizer:
        return self.__column_optim

    @property
    def column_confusion(self) -> np.ndarray:
        return self.__column_confusion

    @property
    def row_confusion(self) -> np.ndarray:
        return self.confusion_matrix

    @property
    def button_confusion(self) -> np.ndarray:
        return self.__button_confusion

    def train(self, train_dataset, neural_net, optimizer):
        row_data_loader, column_data_loader = self._get_data_and_labels(train_dataset)
        trainer = TwoStepTrainer(row_data_loader, column_data_loader, neural_net,
                                 self.column_neural_net, optimizer, self.column_optim, self.criterion,
                                 self.knitted_component, self.learning_def, self.column_learning_def, self.parameters,
                                 self.writer, self.model_path)

        # save the trained models and their paths as well
        torch.save(neural_net, self.row_model_path)
        torch.save(self.column_neural_net, self.column_model_path)

        return trainer.row_loss, trainer.row_accuracy, trainer.column_loss, trainer.column_accuracy, \
               trainer.button_accuracy

    # evaluate the learning created during training on the validation dataset
    def val(self, val_dataset, model_path=None):
        row_data_loader, column_data_loader = self._get_data_and_labels(val_dataset)

        if model_path is None:
            row_model = torch.load(self.row_model_path)
        else:
            row_model = torch.load(model_path)

        column_model = torch.load(self.column_model_path)

        evaluator = TwoStepEvaluator(row_data_loader, column_data_loader, row_model, column_model,
                                     self.criterion, self.knitted_component, self.row_confusion,
                                     self.column_confusion, self.button_confusion,
                                     self.learning_def, self.parameters, self.writer)

        row_fold_accuracy = evaluator.row_accuracy
        column_fold_accuracy = evaluator.column_accuracy
        button_fold_accuracy = evaluator.button_accuracy

        self.row_fold_accuracies.append(row_fold_accuracy)
        self.col_fold_accuracies.append(column_fold_accuracy)
        self.button_fold_accuracies.append(button_fold_accuracy)
        # self.all_val_accuracies.append(fold_accuracy)

        return evaluator.row_loss, evaluator.row_accuracy, evaluator.column_loss,evaluator.column_accuracy, \
               evaluator.button_accuracy

    def _specific_train_val(self, balanced_train, balanced_val, neural_net, optimizer):
        epoch_train_row_losses = []
        epoch_train_row_accuracies = []
        epoch_train_column_losses = []
        epoch_train_column_accuracies = []
        epoch_train_button_accuracies = []

        epoch_val_row_losses = []
        epoch_val_row_accuracies = []
        epoch_val_column_losses = []
        epoch_val_column_accuracies = []
        epoch_val_button_accuracies = []

        train_time_s = 0
        val_time_s = 0
        for epoch in range(1, self.learning_def.num_epochs + 1):

            train_start = time.time()
            train_row_loss, train_row_accuracy, train_column_loss, \
            train_column_accuracy, train_button_accuracy = self.train(balanced_train, neural_net, optimizer)

            train_time_diff = utils.time_diff(train_start)
            train_time_s += train_time_diff

            # start validating the learning
            val_start = time.time()
            val_row_loss, val_row_accuracy, val_column_loss, val_column_accuracy, val_button_accuracy = \
                self.val(balanced_val)

            val_time_diff = utils.time_diff(val_start)
            val_time_s += val_time_diff

            # Print epoch number, loss, accuracy, name and guess
            print_every = 10
            if epoch % print_every == 0:
                print(
                    f"Epoch {epoch}: \n"
                    
                    f"Train Row Loss: {(train_row_loss / epoch):.5f}    Train Row Accuracy:"
                    f" {train_row_accuracy:.3f}  Train Column Loss: {(train_column_loss / epoch):.5f}    Train Column "
                    f"Accuracy: {train_column_accuracy:.3f}     Train Button Accuracy: {train_button_accuracy:.3f}  \n"
                    
                    f"Val Row Loss: {(val_row_loss / epoch):.5f}      Val Row Accuracy: {val_row_accuracy: .3f}     "
                    f"Val Column Loss: {(val_column_loss / epoch):.5f}    Val Column Accuracy: "
                    f"{val_column_accuracy:.3f}     Val Button Accuracy: {val_button_accuracy:.3f}     \n")

                # print(f"Train Row Accuracies: {self._all_epoch_train_row_accuracies}\n")
                # print(f"Train Button Accuracies: {self._all_epoch_train_button_accuracies}\n")
                # print(f"Val Row Accuracies: {self._all_epoch_val_row_accuracies}\n")
                # print(f"Val Button Accuracies: {self._all_epoch_val_button_accuracies}\n\n")

            # Add current loss avg to list of losses
            epoch_train_row_losses.append(train_row_loss / epoch)
            epoch_val_row_losses.append(val_row_loss / epoch)
            epoch_train_column_losses.append(train_column_loss / epoch)
            epoch_val_column_losses.append(val_column_loss / epoch)

            epoch_train_row_accuracies.append(train_row_accuracy)
            epoch_val_row_accuracies.append(val_row_accuracy)
            epoch_train_column_accuracies.append(train_column_accuracy)
            epoch_val_column_accuracies.append(val_column_accuracy)
            epoch_train_button_accuracies.append(train_button_accuracy)
            epoch_val_button_accuracies.append(val_button_accuracy)

        self.__all_epoch_train_row_accuracies.append(epoch_train_row_accuracies)
        self.__all_epoch_train_row_losses.append(epoch_train_row_losses)
        self.__all_epoch_train_col_accuracies.append(epoch_train_column_accuracies)
        self.__all_epoch_train_col_losses.append(epoch_train_column_losses)
        self.__all_epoch_train_button_accuracies.append(epoch_train_button_accuracies)

        self.__all_epoch_val_row_accuracies.append(epoch_val_row_accuracies)
        self.__all_epoch_val_row_losses.append(epoch_val_row_losses)
        self.__all_epoch_val_col_accuracies.append(epoch_val_column_accuracies)
        self.__all_epoch_val_col_losses.append(epoch_val_column_losses)
        self.__all_epoch_val_button_accuracies.append(epoch_val_button_accuracies)

        self.train_time = utils.time_s_to_str(train_time_s)
        self.val_time = utils.time_s_to_str(val_time_s)
        # print("Exiting....")
        # sys.exit()

    def _store_specific_results(self):
        super()._store_specific_results()

        return

    def _get_data_and_labels(self, subj_dataset):
        if self.learning_def.nn_name == "MLP":
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

        row_labels, column_labels = self.knitted_component.get_row_column_labels(labels)
        row_labels = torch.from_numpy(row_labels)
        column_labels = torch.from_numpy(column_labels)

        # standardized_data = self.standardize(data)
        row_tensor_dataset = TensorDataset(standardized_data, row_labels)
        column_tensor_dataset = TensorDataset(standardized_data, column_labels)

        # DO NOT SHUFFLE either one !!!!! Later code relies on maintaining order
        row_data_loader = DataLoader(row_tensor_dataset, batch_size=self.learning_def.batch_size,
                                     num_workers=self.parameters.num_threads, shuffle=False, pin_memory=False)

        column_data_loader = DataLoader(column_tensor_dataset, batch_size=self.learning_def.batch_size,
                                     num_workers=self.parameters.num_threads, shuffle=False, pin_memory=False)

        return row_data_loader, column_data_loader

    def _specific_train_only(self, balanced_train, neural_net, optimizer):
        epoch_train_row_losses = []
        epoch_train_row_accuracies = []
        epoch_train_column_losses = []
        epoch_train_column_accuracies = []
        epoch_train_button_accuracies = []
        train_time_s = 0

        print(f"\nNetwork Architecture: {neural_net}\n")
        for epoch in range(1, self.learning_def.num_epochs + 1):

            train_start = time.time()
            row_loss, row_accuracy, column_loss, column_accuracy, button_accuracy = self.train(balanced_train,
                                                                                               neural_net, optimizer)
            train_time_diff = utils.time_diff(train_start)
            train_time_s += train_time_diff

            print_every = 10
            if epoch % print_every == 0:
                print(
                    f"Epoch {epoch}\n: Train Row Loss: {(row_loss / epoch):.5f}    Train Row Accuracy:"
                    f" {row_accuracy:.3f}     Train Column Loss: {(column_loss / epoch):.5f}"
                    f"    Train Row Accuracy: {column_accuracy:.3f}     Train Button Accuracy: {button_accuracy:.3f}\n")

            # Add current loss avg to list of losses
            epoch_train_row_losses.append(row_loss / epoch)
            epoch_train_row_accuracies.append(row_accuracy)
            epoch_train_column_losses.append(column_loss / epoch)
            epoch_train_column_accuracies.append(column_accuracy / epoch)
            epoch_train_button_accuracies.append(button_accuracy)

            self.__all_epoch_train_row_accuracies.append(epoch_train_row_accuracies)
            self.__all_epoch_train_row_losses.append(epoch_train_row_losses)
            self.__all_epoch_train_col_accuracies.append(epoch_train_column_accuracies)
            self.__all_epoch_train_col_losses.append(epoch_train_column_losses)
            self.__all_epoch_train_button_accuracies.append(epoch_train_button_accuracies)

            self.train_time = utils.time_s_to_str(train_time_s)
