"""
Created: 1/7/20
© Denisa Qori McDonald 2020 All Rights Reserved
"""
import time
from typing import List

import numpy as np
import torch
from torch.optim import Optimizer
from torch.utils.data import DataLoader, TensorDataset

from os.path import join
import BioHCI.helpers.type_aliases as types
import BioHCI.helpers.utilities as utils
from BioHCI.architectures.abstract_neural_net import AbstractNeuralNetwork
from BioHCI.data.data_splitter import DataSplitter
from BioHCI.data_processing.category_balancer import CategoryBalancer
from BioHCI.data_processing.feature_constructor import FeatureConstructor
from BioHCI.definitions.neural_net_def import NeuralNetworkDefinition
from BioHCI.definitions.study_parameters import StudyParameters
from BioHCI.knitted_components.knitted_component import KnittedComponent
from BioHCI.learning.nn_cross_validator import NNCrossValidator
from BioHCI.learning.two_step_evaluator import TwoStepEvaluator
from BioHCI.learning.two_step_trainer import TwoStepTrainer


class KnittingCrossValidator(NNCrossValidator):

    def __init__(self, subject_dict: types.subj_dataset, data_splitter: DataSplitter, feature_constructor:
    FeatureConstructor, category_balancer: CategoryBalancer, neural_net: AbstractNeuralNetwork, secondary_neural_net:
    AbstractNeuralNetwork, parameters: StudyParameters, learning_def: NeuralNetworkDefinition,
                 secondary_learning_def: NeuralNetworkDefinition, all_categories: List[str],
                 knitted_component: KnittedComponent, extra_model_name: str = ""):

        self.__knitted_component = knitted_component
        assert (parameters.neural_net is True), "In StudyParameters, neural_net is set to False and you are " \
                                                "trying to instantiate a NNCrossValidator object!"
        self.__secondary_nn = secondary_neural_net
        self.__secondary_learning_def = secondary_learning_def
        self.__secondary_confusion = np.zeros((knitted_component.num_buttons, knitted_component.num_buttons))

        # TODO: add a list of models created and make sure that things are saved properly - might be challenging
        self.__nn_ls = []
        self.__optim_ls = []
        self.__secondary_model_ls = []              # gets populated as models are produced
        self.__secondary_confusion_ls = []
        self.__populate_secondary_nn_ls()
        self.__populate_secondary_optim_ls()

        self.__populate_secondary_confusion_ls()
        super(KnittingCrossValidator, self).__init__(subject_dict, data_splitter, feature_constructor,
                                                     category_balancer, neural_net, parameters, learning_def,
                                                     all_categories, extra_model_name)

    @property
    def knitted_component(self) -> KnittedComponent:
        return self.__knitted_component

    @property
    def secondary_learning_def(self) -> NeuralNetworkDefinition:
        return self.__secondary_learning_def

    @property
    def secondary_neural_net_ls(self) -> List[AbstractNeuralNetwork]:
        return self.__nn_ls

    @property
    def secondary_model_ls(self) -> List[AbstractNeuralNetwork]:
        return self.__secondary_model_ls

    @property
    def secondary_optim_ls(self) -> List[Optimizer]:
        return self.__optim_ls

    @property
    def secondary_confusion_ls(self) -> List[np.ndarray]:
        return self.__secondary_confusion_ls

    def __populate_secondary_nn_ls(self) -> None:
        for _ in range(0, self.knitted_component.num_rows):
            self.__nn_ls.append(self.__secondary_nn)

    def __populate_secondary_optim_ls(self) -> None:
        for nn in self.secondary_neural_net_ls:
            optim = torch.optim.Adam(nn.parameters(), lr=self.secondary_learning_def.learning_rate)
            self.__optim_ls.append(optim)

    def __populate_secondary_confusion_ls(self) -> None:
        for _ in range(0, self.knitted_component.num_rows):
            self.__secondary_confusion_ls.append(self.__secondary_confusion)

    def train(self, train_dataset):
        row_data_loader, button_data_loader = self._get_data_and_labels(train_dataset)
        trainer = TwoStepTrainer(row_data_loader, button_data_loader, self.neural_net, self.secondary_neural_net_ls,
                    self.optimizer, self.secondary_optim_ls, self.criterion, self.knitted_component, self.learning_def,
                    self.secondary_learning_def, self.parameters, self.writer, self.model_path)

        # save the trained models and their paths as well
        for r, nn in enumerate(self.secondary_neural_net_ls):
            model_name = self.general_name + "_r" + str(r) + ".pt"
            model_path = join(self.model_dir, model_name)
            torch.save(nn, model_path)
            self.secondary_model_ls.append(model_path)

        return trainer.row_loss, trainer.row_accuracy, trainer.button_loss, trainer.button_accuracy

    # evaluate the learning created during training on the validation dataset
    def val(self, val_dataset, model_path=None):
        row_data_loader, button_data_loader = self._get_data_and_labels(val_dataset)

        if model_path is None:
            model_to_eval = torch.load(self.model_path)
        else:
            model_to_eval = torch.load(model_path)

        evaluator = TwoStepEvaluator(row_data_loader, button_data_loader, model_to_eval, self.secondary_model_ls,
                    self.criterion, self.knitted_component, self.confusion_matrix, self.secondary_confusion_ls,
                                     self.learning_def, self.parameters, self.writer)

        fold_accuracy = evaluator.accuracy
        self.all_val_accuracies.append(fold_accuracy)

        return evaluator.loss, evaluator.accuracy

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

        row_labels = self.knitted_component.get_row_labels(labels)
        row_labels = torch.from_numpy(row_labels)
        button_labels = torch.from_numpy(labels)

        standardized_data = self.standardize(data)
        row_tensor_dataset = TensorDataset(standardized_data, row_labels)
        button_tensor_dataset = TensorDataset(standardized_data, button_labels)

        # DO NOT SHUFFLE either one !!!!! Later code relies on maintaining order
        row_data_loader = DataLoader(row_tensor_dataset, batch_size=self.learning_def.batch_size,
                                     num_workers=self.parameters.num_threads, shuffle=False, pin_memory=True)

        button_data_loader = DataLoader(button_tensor_dataset, batch_size=self.learning_def.batch_size,
                                        num_workers=self.parameters.num_threads, shuffle=False, pin_memory=True)

        return row_data_loader, button_data_loader

    def _specific_train_only(self, balanced_train):
        epoch_train_row_losses = []
        epoch_train_row_accuracies = []
        epoch_train_button_losses = []
        epoch_train_button_accuracies = []
        train_time_s = 0

        print(f"\nNetwork Architecture: {self.neural_net}\n")
        for epoch in range(1, self.learning_def.num_epochs + 1):

            train_start = time.time()
            current_train_row_loss, current_train_row_accuracy, current_train_button_loss, \
            current_train_button_accuracy = self.train(balanced_train)
            train_time_diff = utils.time_diff(train_start)
            train_time_s += train_time_diff

            self.writer.add_scalar('Train Row Loss', current_train_row_loss, epoch)
            self.writer.add_scalar('Train Row Accuracy', current_train_row_accuracy, epoch)

            self.writer.add_scalar('Train Button Loss', current_train_button_loss, epoch)
            self.writer.add_scalar('Train Button Accuracy', current_train_button_accuracy, epoch)

            # Print epoch number, loss, accuracy, name and guess
            print_every = 10
            if epoch % print_every == 0:
                print(
                    f"Epoch {epoch}:    Train Row Loss: {(current_train_row_loss / epoch):.5f}    Train Row Accuracy:"
                    f" {current_train_row_accuracy:.3f}     Train Button Loss: {(current_train_button_loss / epoch):.5f}"
                    f"    Train Button Accuracy: {current_train_button_accuracy:.3f} ")

            # Add current loss avg to list of losses
            epoch_train_row_losses.append(current_train_row_loss / epoch)
            epoch_train_row_accuracies.append(current_train_row_accuracy)
            epoch_train_button_losses.append(current_train_button_loss / epoch)
            epoch_train_button_accuracies.append(current_train_button_accuracy)

            self.writer.add_scalar('Train Avg Loss', current_train_button_loss / epoch, epoch)

        self.__all_epoch_train_accuracies.append(epoch_train_button_accuracies)
        self.__all_epoch_train_losses.append(epoch_train_button_losses)

        self.train_time = utils.time_s_to_str(train_time_s)
