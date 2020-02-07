import time
from typing import List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import BioHCI.helpers.type_aliases as types
from BioHCI.data.data_splitter import DataSplitter
from BioHCI.data_processing.category_balancer import CategoryBalancer
from BioHCI.data_processing.feature_constructor import FeatureConstructor
from BioHCI.data_processing.keypoint_description.desc_type import DescType
from BioHCI.data_processing.keypoint_description.descriptor_computer import DescriptorComputer
from BioHCI.data_processing.keypoint_description.sequence_length import SeqLen
from BioHCI.definitions.neural_net_def import NeuralNetworkDefinition
from BioHCI.definitions.study_parameters import StudyParameters
from BioHCI.helpers import utilities as utils
from BioHCI.knitted_components.knitted_component import KnittedComponent
from BioHCI.learning.cross_validator import CrossValidator
from BioHCI.learning.evaluator import Evaluator
from BioHCI.learning.trainer import Trainer


class NNCrossValidator(CrossValidator):

    def __init__(self, subject_dict: types.subj_dataset, data_splitter: DataSplitter,
                 feature_constructor: FeatureConstructor, category_balancer: CategoryBalancer,
                 parameters: StudyParameters, learning_def: NeuralNetworkDefinition, all_categories: List[str],
                 knitted_component: KnittedComponent, extra_model_name: str = ""):
        assert (parameters.neural_net is True), "In StudyParameters, neural_net is set to False and you are " \
                                                "trying to instantiate a NNCrossValidator object!"
        # this list contains lists of accuracies for each epoch. There will be self._num_folds lists of _num_epochs
        # elements in this list after all training is done
        self.__learning_def = learning_def
        self.knitted_component = knitted_component

        super(NNCrossValidator, self).__init__(subject_dict, data_splitter, feature_constructor, category_balancer,
                                               parameters, learning_def, all_categories, extra_model_name)

        if parameters.classification:
            # the negative log likelihood loss function - useful to train classification problems with C classes
            self.__criterion = nn.NLLLoss()
        else:
            self.__criterion = nn.SmoothL1Loss()

    @property
    def criterion(self):
        return self.__criterion

    @property
    def learning_def(self) -> NeuralNetworkDefinition:
        return self.__learning_def

    # implement the abstract method from the parent class CrossValidator; returns a dataset with labels wrapped in
    # the PyTorch DataLoader format
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

        labels = torch.from_numpy(labels)
        # the tensor_dataset is a tuple of TensorDataset type, containing a tensor with data (train or val),
        # and one with labels (train or val respectively)

        row_labels, column_labels = self.knitted_component.get_row_column_labels(labels)
        row_labels = torch.from_numpy(row_labels)

        standardized_data = self.standardize(data)
        tensor_dataset = TensorDataset(standardized_data, row_labels)

        data_loader = DataLoader(tensor_dataset, batch_size=self.learning_def.batch_size,
                                 num_workers=self.parameters.num_threads, shuffle=False, pin_memory=False)

        return data_loader

    # implement the abstract method from the parent class CrossValidator; it is called for each fold in
    # cross-validation and after it trains for that fold, it appends the calculated losses and accuracies for each
    # epoch to the respective list in the CrossValidator object standout
    def train(self, train_dataset, neural_net, optimizer):
        train_data_loader = self._get_data_and_labels(train_dataset)
        trainer = Trainer(train_data_loader, neural_net, optimizer, self.criterion,
                          self.learning_def, self.parameters, self.writer, self.model_path)

        return trainer.loss, trainer.accuracy

    # evaluate the learning created during training on the validation dataset
    def val(self, val_dataset, model_path=None):
        val_data_loader = self._get_data_and_labels(val_dataset)
        # label_dict = self.compute_label_msd_dict(val_dataset)

        if model_path is None:
            model_to_eval = torch.load(self.model_path)
        else:
            model_to_eval = torch.load(model_path)

        evaluator = Evaluator(val_data_loader, model_to_eval, self.criterion, self.confusion_matrix, self.learning_def,
                              self.parameters, self.writer)

        fold_accuracy = evaluator.accuracy
        # self.all_val_accuracies.append(fold_accuracy)

        return evaluator.loss, evaluator.accuracy

    def compute_label_msd_dict(self, val_subject_dict: types.subj_dataset):
        descriptor_computer = DescriptorComputer(DescType.MSD, val_subject_dict, self.parameters,
                                                 seq_len=SeqLen.ExtendEdge, extra_name="_button")

        descriptors = descriptor_computer.dataset_descriptors
        all_data, all_labels = self.get_all_subj_data(descriptors)
        labels = utils.convert_categories(self.category_map, all_labels)

        new_index_sets = [np.argwhere(i[0] == labels) for i in np.array(np.unique(labels, return_counts=True)).T]

        # labels, idx_start, count = np.unique(all_labels, return_index=True, return_counts=True)
        # res = np.split(all_labels, idx_start[1:])
        # labels = labels[count > 1]
        # res = filter(lambda x: x.size > 1, res)

        return None

    def _specific_train_val(self, balanced_train, balanced_val, neural_net, optimizer):
        train_time_s = 0
        val_time_s = 0

        train_loss = None
        train_accuracy = None
        val_loss = None
        val_accuracy = None

        for epoch in range(1, self.learning_def.num_epochs + 1):

            train_start = time.time()
            current_train_loss, train_accuracy = self.train(balanced_train, neural_net, optimizer)
            train_time_diff = utils.time_diff(train_start)
            train_time_s += train_time_diff

            self.writer.add_scalar('Train Loss', current_train_loss, epoch)
            self.writer.add_scalar('Train Accuracy', train_accuracy, epoch)

            # after each epoch the new trained model
            torch.save(neural_net, self.model_path)

            # start validating the learning
            val_start = time.time()
            current_val_loss, val_accuracy = self.val(balanced_val)
            val_time_diff = utils.time_diff(val_start)
            val_time_s += val_time_diff

            self.writer.add_scalar('Val Loss', current_val_loss, epoch)
            self.writer.add_scalar('Val Accuracy', val_accuracy, epoch)

            train_loss = current_train_loss / epoch
            val_loss = current_val_loss / epoch

            # Print epoch number, loss, accuracy, name and guess
            print_every = 10
            if epoch % print_every == 0:
                self.result_logger.info(
                    f"Epoch {epoch}: Train Loss: {train_loss :.5f}    Train Accuracy:"
                    f" {train_accuracy:.3f}    Val Loss: {val_loss :.5f}    Val Accuracy: {val_accuracy:.3f}")

            self.writer.add_scalar('Train Avg Loss', train_loss, epoch)
            self.writer.add_scalar('Val Avg Loss', val_loss, epoch)

        train_time = utils.time_s_to_str(train_time_s)
        val_time = utils.time_s_to_str(val_time_s)

        self.result_logger.info(f"\nTrain time (over last cross-validation pass): {train_time}")
        self.result_logger.info(f"Test time (over last cross-validation pass): {val_time}")

        return train_loss, train_accuracy, val_loss, val_accuracy

    def _specific_train_only(self, balanced_train, neural_net, optimizer):
        train_time_s = 0

        for epoch in range(1, self.learning_def.num_epochs + 1):
            train_start = time.time()
            current_train_loss, current_train_accuracy = self.train(balanced_train, neural_net, optimizer)
            train_time_diff = utils.time_diff(train_start)
            train_time_s += train_time_diff

            self.writer.add_scalar('Train Loss', current_train_loss, epoch)
            self.writer.add_scalar('Train Accuracy', current_train_accuracy, epoch)

            # Print epoch number, loss, accuracy, name and guess
            print_every = 10
            if epoch % print_every == 0:
                self.result_logger.info(
                    f"Epoch {epoch}:    Train Loss: {(current_train_loss / epoch):.5f}    Train Accuracy:"
                    f" {current_train_accuracy:.3f}")

            self.writer.add_scalar('Train Avg Loss', current_train_loss / epoch, epoch)

        train_time = utils.time_s_to_str(train_time_s)
        self.result_logger.info(f"\nTrain time (over last cross-validation pass): {train_time}")

    # does not actually use epochs
    def _specific_eval_only(self, balanced_val, model_path=None):
        val_time_s = 0

        # start validating the learning
        val_start = time.time()
        current_val_loss, current_val_accuracy = self.val(balanced_val, model_path)
        val_time_diff = utils.time_diff(val_start)
        val_time_s += val_time_diff

        self.writer.add_scalar('Val Loss', current_val_loss)
        self.writer.add_scalar('Val Accuracy', current_val_accuracy)

        val_time = utils.time_s_to_str(val_time_s)
        self.result_logger.info(f"\nTest time (over last cross-validation pass): {val_time}")

