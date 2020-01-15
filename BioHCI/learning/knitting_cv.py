"""
Created: 1/7/20
© Denisa Qori McDonald 2020 All Rights Reserved
"""
from typing import List

import numpy as np
import torch
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
from BioHCI.learning.two_step_evaluator import TwoStepEvaluator
from BioHCI.learning.two_step_trainer import TwoStepTrainer
from BioHCI.learning.nn_cross_validator import NNCrossValidator


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
        self.__nn_array = []
        self.__populate_secondary_nn_ls()
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
        return self.__nn_array

    def __populate_secondary_nn_ls(self) -> None:
        for _ in range(0, self.knitted_component.num_rows):
            self.__nn_array.append(self.__secondary_nn)

    def train(self, train_dataset):
        row_data_loader, button_data_loader = self._get_data_and_labels(train_dataset)
        trainer = TwoStepTrainer(row_data_loader, button_data_loader, self.neural_net, self.secondary_neural_net_ls,
                                 self.optimizer, self.criterion, self.knitted_component, self.learning_def,
                                 self.secondary_learning_def, self.parameters, self.writer, self.model_path)

        return trainer.loss, trainer.accuracy

    # evaluate the learning created during training on the validation dataset
    def val(self, val_dataset, model_path=None):
        row_data_loader, button_data_loader = self._get_data_and_labels(val_dataset)

        if model_path is None:
            model_to_eval = torch.load(self.model_path)
        else:
            model_to_eval = torch.load(model_path)

        evaluator = TwoStepEvaluator(row_data_loader, model_to_eval, self.criterion, self.knitted_component,
                                     self.confusion_matrix, self.learning_def, self.parameters, self.writer)

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
