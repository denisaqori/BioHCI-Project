from BioHCI.data.data_splitter import DataSplitter
from BioHCI.data_processing.category_balancer import CategoryBalancer
from BioHCI.data_processing.feature_constructor import FeatureConstructor
from BioHCI.definitions.neural_net_def import NeuralNetworkDefinition
from BioHCI.definitions.study_parameters import StudyParameters
from BioHCI.learning.cross_validator import CrossValidator
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from BioHCI.learning.trainer import Trainer
from BioHCI.learning.evaluator import Evaluator
from BioHCI.helpers import utilities as util

import BioHCI.helpers.type_aliases as types
from typing import List

import torch.nn as nn
import torch
import os


class NNCrossValidator(CrossValidator):

    def __init__(self, subject_dict: types.subj_dataset, data_splitter: DataSplitter, feature_constructor:
    FeatureConstructor, category_balancer: CategoryBalancer, model, parameter: StudyParameters, learning_def: \
            NeuralNetworkDefinition, all_categories:
    List[
        str]):
        # this list contains lists of accur   acies for each epoch. There will be self._num_folds lists of _num_epochs
        # elements in this list after all training is done
        self.__all_epoch_train_accuracies = []

        super(NNCrossValidator, self).__init__(subject_dict, data_splitter, feature_constructor, category_balancer,
                                               model, parameter, learning_def, all_categories)
        # the stochastic gradient descent function to update weights a        self.perform_cross_validation()nd biases
        self.__optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_def.learning_rate)
        # the negative log likelihood loss function - useful to train classification problems with C classes
        self.__criterion = nn.NLLLoss()

        assert (parameter.neural_net is True), "In StudyParameters, neural_net is set to False and you are " \
                                               "trying to instantiate a NNCrossValidator object!"

    @property
    def all_epoch_train_accuracies(self) -> List[float]:
        return self.__all_epoch_train_accuracies

    @property
    def optimizer(self):
        return self.__optimizer

    @property
    def criterion(self):
        return self.__criterion

    # implement the abstract method from the parent class CrossValidator; returns a dataset with labels wrapped in
    # the PyTorch DataLoader format
    def _get_data_and_labels(self, subj_dataset):
        data, cat = self.mix_subj_chunks(subj_dataset)

        # convert numpy ndarray to PyTorch tensor
        data = torch.from_numpy(data)
        # convert categories from string to integer
        int_cat = self.all_int_categories
        cat = torch.from_numpy(int_cat)

        # the tensor_dataset is a tuple of TensorDataset type, containing a tensor with data (train or val),
        # and one with labels (train or val respectively)
        tensor_dataset = TensorDataset(data, cat)

        print("Using the PyTorch DataLoader to load the training data (shuffled) with: \nbatch size = ",
              self.learning_def.batch_size, " & number of threads = ", self.parameters.num_threads)

        data_loader = DataLoader(tensor_dataset, batch_size=self.learning_def.batch_size,
                                 num_workers=self.parameters.num_threads, shuffle=False, pin_memory=True)

        return data_loader

    # implement the abstract method from the parent class CrossValidator; it is called for each fold in
    # cross-validation and after it trains for that fold, it appends the calculated losses and accuracies for each
    # epoch to the respective list in the CrossValidator object
    def train(self, train_dataset, summary_writer):
        train_data_loader = self._get_data_and_labels(train_dataset)
        trainer = Trainer(train_data_loader, self.model, self.optimizer, self.criterion, self.all_int_categories,
                          self.learning_def, self.parameters, summary_writer)

        # get the loss over all epochs for this cv-fold and append it to the list
        self.all_train_losses.append(trainer.epoch_losses())
        print("Train Epoch Losses: ", trainer.epoch_losses())

        # accuracies for each epoch and each fold are added to the list that belongs only to this class
        # "_all_epoch_train_accuracies". The last accuracy of each train epoch is added to the list
        # "_all_train_accuracies, belonging more generally to the parent class
        self.all_train_accuracies.append(trainer.epoch_accuracies()[-1])
        self.all_epoch_train_accuracies.append(trainer.epoch_accuracies())
        print("Train Epoch Accuracies: ", trainer.epoch_accuracies())

    # evaluate the learning created during training on the validation dataset
    def val(self, val_dataset, summary_writer):
        val_data_loader = self._get_data_and_labels(val_dataset)

        # this is the architectures produces by training over the other folds
        model_name = self.parameters.study_name + "-" + self.model.name + "-batch-" \
                     + str(self.learning_def.batch_size) + "-seqSize-" \
                     + str(self.parameters.samples_per_chunk) + ".pt"

        saved_models_root = util.get_root_path("saved_objects")
        model_to_eval = torch.load(os.path.join(saved_models_root, model_name))

        evaluator = Evaluator(val_data_loader, model_to_eval, self.all_int_categories, self.confusion_matrix,
                              self.learning_def, summary_writer)

        fold_accuracy = evaluator.get_accuracy()
        self.all_val_accuracies.append(fold_accuracy)
