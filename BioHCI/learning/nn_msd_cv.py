"""
Created: 2/10/20
© Denisa Qori McDonald 2020 All Rights Reserved
"""
from typing import List

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

import BioHCI.helpers.type_aliases as types
import BioHCI.helpers.utilities as utils
from BioHCI.data.data_splitter import DataSplitter
from BioHCI.data_processing.category_balancer import CategoryBalancer
from BioHCI.data_processing.feature_constructor import FeatureConstructor
from BioHCI.data_processing.keypoint_description.desc_type import DescType
from BioHCI.data_processing.keypoint_description.sequence_length import SeqLen
from BioHCI.definitions.neural_net_def import NeuralNetworkDefinition
from BioHCI.definitions.study_parameters import StudyParameters
from BioHCI.knitted_components.knitted_component import KnittedComponent
from BioHCI.learning.nn_cross_validator import NNCrossValidator
from BioHCI.learning.nn_msd_evaluator import NN_MSD_Evaluator


class NN_MSD_CrossValidator(NNCrossValidator):
    def __init__(self, subject_dict: types.subj_dataset, data_splitter: DataSplitter,
                 feature_constructor: FeatureConstructor, category_balancer: CategoryBalancer,
                 parameters: StudyParameters, learning_def: NeuralNetworkDefinition, all_categories: List[str],
                 knitted_component: KnittedComponent, extra_model_name: str = ""):

        super(NN_MSD_CrossValidator, self).__init__(subject_dict, data_splitter, feature_constructor,
                                                    category_balancer, parameters, learning_def, all_categories,
                                                    extra_model_name)
        self.__knitted_component = knitted_component

    @property
    def knitted_component(self):
        return self.__knitted_component

    def get_train_dataloader(self, subj_dataset):
        if self.learning_def.nn_name == "MLP":
            data, cat = self.get_all_subj_data(subj_dataset, seq=False)
        else:
            data, cat = self.get_all_subj_data(subj_dataset)

        # convert numpy ndarray to PyTorch tensor
        np_data = np.asarray(data, dtype=np.float32)
        data = torch.from_numpy(np_data)

        # convert categories from string to integer
        labels = utils.convert_categories(self.category_map, cat)

        button_labels = torch.from_numpy(labels)
        # the tensor_dataset is a tuple of TensorDataset type, containing a tensor with data (train or val),
        # and one with labels (train or val respectively)

        row_labels, column_labels = self.knitted_component.get_row_column_labels(button_labels)
        row_labels = torch.from_numpy(row_labels)

        standardized_data = self.standardize(data)
        # Tensor Dataset built with two labels on the same data !!!!!!!!!!
        dataset = TensorDataset(standardized_data, row_labels)
        data_loader = DataLoader(dataset, batch_size=self.learning_def.batch_size,
                                 num_workers=self.parameters.num_threads, shuffle=False, pin_memory=False)

        return data_loader

    # implement the abstract method from the parent class CrossValidator; returns a dataset with labels wrapped in
    # the PyTorch DataLoader format
    def get_val_dataloader(self, subj_dataset):
        if self.learning_def.nn_name == "MLP":
            data, cat = self.get_all_subj_data(subj_dataset, seq=False)
        else:
            data, cat = self.get_all_subj_data(subj_dataset)

        # convert numpy ndarray to PyTorch tensor
        np_data = np.asarray(data, dtype=np.float32)
        data = torch.from_numpy(np_data)

        # convert categories from string to integer
        labels = utils.convert_categories(self.category_map, cat)

        button_labels = torch.from_numpy(labels)
        # the tensor_dataset is a tuple of TensorDataset type, containing a tensor with data (train or val),
        # and one with labels (train or val respectively)

        row_labels, column_labels = self.knitted_component.get_row_column_labels(button_labels)
        row_labels = torch.from_numpy(row_labels)

        standardized_data = self.standardize(data)
        # Tensor Dataset built with two labels on the same data !!!!!!!!!!
        dataset = TensorDataset(standardized_data, row_labels, button_labels)
        data_loader = DataLoader(dataset, batch_size=self.learning_def.batch_size,
                                 num_workers=self.parameters.num_threads, shuffle=False, pin_memory=False)

        return data_loader

    # def train(self, train_dataset, neural_net, optimizer):
    #     train_loss, train_accuracy = NNCrossValidator.train(self, train_dataset, neural_net, optimizer)
    #     self.__msd_train_dict = self.compute_label_msd_dict(train_dataset)
    #     return train_loss, train_accuracy

    # evaluate the learning created during training on the validation dataset
    def val(self, val_dataset, model_path=None):
        val_data_loader = self.get_val_dataloader(val_dataset)

        if model_path is None:
            model_to_eval = torch.load(self.model_path)
        else:
            model_to_eval = torch.load(model_path)

        evaluator = NN_MSD_Evaluator(self.msd_train_dict, self.desc_type, self.seq_len, self.knitted_component,
                                     model_to_eval, self.criterion, self.learning_def, self.parameters, self.writer)

        row_loss, row_accuracy, button_accuracy, button_only_accuracy = evaluator.evaluate(val_data_loader,
                                                                                  self.confusion_matrix)
        self.result_logger.info(f"Button Accuracy: {button_accuracy}    Button Only Accuracy: {button_only_accuracy}")
        return row_loss, row_accuracy
