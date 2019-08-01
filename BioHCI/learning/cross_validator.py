import time
from abc import ABC, abstractmethod
from os.path import join
from typing import List, Tuple, Optional

import numpy as np
import torch
from tensorboardX import SummaryWriter

import BioHCI.helpers.type_aliases as types
import BioHCI.helpers.utilities as utils
from BioHCI.data.data_splitter import DataSplitter
from BioHCI.data_processing.category_balancer import CategoryBalancer
from BioHCI.data_processing.feature_constructor import FeatureConstructor
from BioHCI.definitions.learning_def import LearningDefinition
from BioHCI.definitions.study_parameters import StudyParameters


class CrossValidator(ABC):
    def __init__(self, subject_dict: types.subj_dataset, data_splitter: DataSplitter, feature_constructor:
    FeatureConstructor, category_balancer: CategoryBalancer, model, parameters: StudyParameters, learning_def:
    LearningDefinition, all_categories: List[str]):
        self.__subject_dict = subject_dict
        self.__data_splitter = data_splitter
        self.__feature_constructor = feature_constructor
        self.__category_balancer = category_balancer
        self.__all_categories = all_categories
        self.__all_int_categories = None
        self.__model = model
        self.__learning_def = learning_def
        self.__parameters = parameters
        self.__num_folds = parameters.num_folds

        self.__all_val_accuracies = []
        self.__all_train_accuracies = []
        self.__all_train_losses = []

        # declare variables that will contain time needed to compute these operations
        self.__cv_time = ""
        self.__train_time = ""
        self.__val_time = 0

        # TODO: logger should probably be initialized here too
        tbx_name = parameters.study_name + "/tensorboardX_runs"
        self.__tbx_path = utils.create_dir(join(utils.get_root_path("Results"), tbx_name))

        self.__writer = SummaryWriter(self.tbx_path)

        # create a confusion matrix to track correct guesses (accumulated over all folds of the Cross-Validation
        # below
        self.__confusion_matrix = torch.zeros(len(all_categories), len(all_categories))

        # self._confusion_matrix = np.zeros((len(all_categories), len(all_categories)))

    @property
    def subject_dict(self) -> types.subj_dataset:
        return self.__subject_dict

    @property
    def data_splitter(self) -> DataSplitter:
        return self.__data_splitter

    @property
    def feature_constructor(self) -> FeatureConstructor:
        return self.__feature_constructor

    @property
    def category_balancer(self) -> CategoryBalancer:
        return self.__category_balancer

    @property
    def all_categories(self) -> List[str]:
        return self.__all_categories

    @property
    def all_int_categories(self) -> List[int]:
        return self.__all_int_categories

    @all_int_categories.setter
    def all_int_categories(self, categories: Optional[List[int]]):
        self.__all_int_categories = categories

    @property
    def model(self):
        return self.__model

    @property
    def learning_def(self) -> LearningDefinition:
        return self.__learning_def

    @property
    def parameters(self) -> StudyParameters:
        return self.__parameters

    @property
    def num_folds(self) -> int:
        return self.__num_folds

    @property
    def all_val_accuracies(self) -> List[float]:
        return self.__all_val_accuracies

    @property
    def all_train_accuracies(self) -> List[float]:
        return self.__all_train_accuracies

    @property
    def all_train_losses(self) -> List[float]:
        return self.__all_train_losses

    @property
    def cv_time(self) -> str:
        return self.__cv_time

    @cv_time.setter
    def cv_time(self, cv_time: str):
        self.__cv_time = cv_time

    @property
    def train_time(self) -> str:
        return self.__train_time

    @train_time.setter
    def train_time(self, train_time: str):
        self.__train_time = train_time

    @property
    def val_time(self):
        return self.__val_time

    @val_time.setter
    def val_time(self, val_time: str):
        self.__val_time = val_time

    @property
    def tbx_path(self):
        return self.__tbx_path

    @property
    def writer(self):
        return self.__writer

    @property
    def confusion_matrix(self):
        return self.__confusion_matrix

    def perform_cross_validation(self) -> None:
        cv_start = time.time()

        feature_dataset = self.feature_constructor.produce_feature_dataset(self.subject_dict)

        for i in range(0, self.num_folds):
            print("\n\n"
                  "***************************************************************************************************")
            print("Run: ", i)
            train_dataset, val_dataset = self.data_splitter.split_into_folds_features(
                feature_dictionary=feature_dataset, num_folds=self.num_folds, val_index=i)
            # train_dataset, val_dataset = self.data_splitter.split_into_folds_raw(
            #     subject_dictionary=self.subject_dict, num_folds=self.num_folds, val_index=i)

            # train_feature = self.feature_constructor.produce_feature_dataset(train_dataset)
            # val_feature = self.feature_constructor.produce_feature_dataset(val_dataset)

            # balance each dataset individually
            balanced_train = self.category_balancer.balance(train_dataset)
            balanced_val = self.category_balancer.balance(val_dataset)

            # starting training with the above-defined parameters
            train_start = time.time()
            self.train(balanced_train, self.writer)
            self.train_time = utils.time_since(train_start)

            # start validating the learning
            val_start = time.time()
            self.val(balanced_val, self.writer)
            self.val_time = utils.time_since(val_start)

        self.cv_time = utils.time_since(cv_start)

    @abstractmethod
    def _get_data_and_labels(self, python_dataset):
        pass

    @abstractmethod
    def train(self, train_dataset, summary_writer):
        pass

    @abstractmethod
    def val(self, val_dataset, summary_writer):
        pass

    @property
    def avg_train_accuracy(self) -> float:
        """Compute the average of train accuracy of each fold's last epoch."""

        # return the average by dividing the sum by the number of folds (= number of accuracies added)
        avg_accuracy = sum(self.all_train_accuracies) / float(len(self.all_train_accuracies))
        print("\nAverage train accuracy over", self.num_folds, "is", avg_accuracy)
        return avg_accuracy

    @property
    def avg_val_accuracy(self) -> float:
        avg_accuracy = sum(self.all_val_accuracies) / float(len(self.all_val_accuracies))
        print("\nAverage val accuracy over", self.num_folds, "is", avg_accuracy)
        return avg_accuracy

    @property
    def avg_train_losses(self) -> List[float]:
        avg_losses = []
        for i in range(self.learning_def.num_epochs):
            epoch_loss = 0
            for j, loss_list in enumerate(self.all_train_losses):
                epoch_loss = epoch_loss + loss_list[i]
            avg_losses.append(epoch_loss / self.num_folds)
        return avg_losses


    def mix_subj_data(self, subj_dict: types.subj_dataset) -> Tuple[List[np.ndarray], List[str]]:
        """
        Creates a dataset of chunks of all subjects with the corresponding categories. At this point the subject data
        is not separated anymore.

        Args:
            subj_dict (dict): a dictionary mapping a subject name to a Subject object

        Returns:
            all_data (ndarray): a 2D numpy array containing the train dataset of shape (instances per sample x number of features)
            all_cat (ndarray): a 1D numpy arrray containing the category labels of all_data, of shape (number of
                chunks).
        """

        # data to stack - subjects end up mixed together in the ultimate dataset
        all_data = []
        # list of all categories to return
        all_cat = []

        for subj_name, subj in subj_dict.items():
            for i, data in enumerate(subj.data):
                all_data.append(data)
                all_cat.append(subj.categories[i])

        return all_data, all_cat
