import os
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
from BioHCI.learning.cross_validator import CrossValidator
from BioHCI.learning.evaluator import Evaluator
from BioHCI.learning.trainer import Trainer
from os.path import join


class NNCrossValidator(CrossValidator):

    def __init__(self, subject_dict: types.subj_dataset, data_splitter: DataSplitter, feature_constructor:
    FeatureConstructor, category_balancer: CategoryBalancer, neural_net: AbstractNeuralNetwork, parameters:
    StudyParameters, learning_def: NeuralNetworkDefinition, all_categories: List[str]):
        # this list contains lists of accur   acies for each epoch. There will be self._num_folds lists of _num_epochs
        # elements in this list after all training is done
        self.__all_epoch_train_accuracies = []

        super(NNCrossValidator, self).__init__(subject_dict, data_splitter, feature_constructor, category_balancer,
                                               neural_net, parameters, learning_def, all_categories)
        # the stochastic gradient descent function to update weights a        self.perform_cross_validation()nd biases
        self.__optimizer = torch.optim.Adam(self.neural_net.parameters(), lr=learning_def.learning_rate)
        # the negative log likelihood loss function - useful to train classification problems with C classes
        self.__criterion = nn.NLLLoss()

        assert (parameters.neural_net is True), "In StudyParameters, neural_net is set to False and you are " \
                                               "trying to instantiate a NNCrossValidator object!"

        self.__model_name = self.__produce_model_name()

        model_subdir = parameters.study_name + "/trained_models"
        self.__saved_model_dir = utils.create_dir(join(utils.get_root_path("saved_objects"), model_subdir))
        # create the full name of the dataset as well, without the path to get there
        self.__model_path = join(self.__saved_model_dir, self.model_name)

        utils.cleanup(self.model_dir, "_test")

    @property
    def all_epoch_train_accuracies(self) -> List[float]:
        return self.__all_epoch_train_accuracies

    @property
    def optimizer(self):
        return self.__optimizer

    @property
    def criterion(self):
        return self.__criterion

    @property
    def model_name(self) -> str:
        return self.__model_name

    @property
    def model_path(self) -> str:
        return self.__model_path

    @property
    def model_dir(self) -> str:
        return self.__saved_model_dir

    # implement the abstract method from the parent class CrossValidator; returns a dataset with labels wrapped in
    # the PyTorch DataLoader format
    def _get_data_and_labels(self, subj_dataset):
        data, cat = self.mix_subj_data(subj_dataset)

        # convert numpy ndarray to PyTorch tensor
        data = torch.from_numpy(np.asarray(data))
        # convert categories from string to integer
        int_cat = utils.convert_categories(self.all_categories, cat)
        # set all the categories of the cross validator
        self.all_int_categories = int_cat
        cat = torch.from_numpy(int_cat)

        # the tensor_dataset is a tuple of TensorDataset type, containing a tensor with data (train or val),
        # and one with labels (train or val respectively)
        tensor_dataset = TensorDataset(data, cat)

        print(f"\n\nUsing the PyTorch DataLoader to load the training data (shuffled) with: \nbatch size = "
              f"{self.learning_def.batch_size} & number of threads = {self.parameters.num_threads}")

        data_loader = DataLoader(tensor_dataset, batch_size=self.learning_def.batch_size,
                                 num_workers=self.parameters.num_threads, shuffle=False, pin_memory=True)

        return data_loader

    # implement the abstract method from the parent class CrossValidator; it is called for each fold in
    # cross-validation and after it trains for that fold, it appends the calculated losses and accuracies for each
    # epoch to the respective list in the CrossValidator objectstandout
    def train(self, train_dataset, summary_writer):
        train_data_loader = self._get_data_and_labels(train_dataset)
        trainer = Trainer(train_data_loader, self.neural_net, self.optimizer, self.criterion, self.all_int_categories,
                          self.learning_def, self.parameters, summary_writer, self.model_path)

        # get the loss over all epochs for this cv-fold and append it to the list
        self.all_epoch_train_losses.append(trainer.epoch_losses)
        print("Train Epoch Losses: ", trainer.epoch_losses)

        # accuracies for each epoch and each fold are added to the list that belongs only to this class
        # "_all_epoch_train_accuracies". The last accuracy of each train epoch is added to the list
        # "_all_train_accuracies, belonging more generally to the parent class
        self.all_train_accuracies.append(trainer.epoch_accuracies[-1])
        self.all_epoch_train_accuracies.append(trainer.epoch_accuracies)
        print("Train Epoch Accuracies: ", trainer.epoch_accuracies)

    # evaluate the learning created during training on the validation dataset
    def val(self, val_dataset, summary_writer):
        val_data_loader = self._get_data_and_labels(val_dataset)
        model_to_eval = torch.load(self.model_path)

        evaluator = Evaluator(val_data_loader, model_to_eval, self.all_int_categories, self.confusion_matrix,
                              self.learning_def, summary_writer)

        fold_accuracy = evaluator.get_accuracy()
        self.all_val_accuracies.append(fold_accuracy)

    def _log_specific_results(self):
        self.result_logger.debug(f"All fold train accuracies (all epochs): {str(self.all_epoch_train_accuracies)}")
        self.result_logger.info(f"All fold train accuracies: {str(self.all_train_accuracies)}")
        self.result_logger.info(f"Average train accuracy: {str(self.avg_train_accuracy)}")
        self.result_logger.info(f"All fold validation accuracies: {str(self.all_val_accuracies)}")
        self.result_logger.info(f"Average validation accuracy: {str(self.avg_val_accuracy)}\n")

    def __produce_model_name(self) -> str:
        # this is the model produced by training over all folds - 1
        name = self.parameters.study_name + "-" + self.neural_net.name + "-batch-" + str(
            self.learning_def.batch_size) + ".pt"
        return name
