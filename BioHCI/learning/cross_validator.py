import logging
import platform
import time
from abc import ABC, abstractmethod
from datetime import datetime
from os.path import join
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from tensorboardX import SummaryWriter

import BioHCI.helpers.type_aliases as types
import BioHCI.helpers.utilities as utils
from BioHCI.architectures.abstract_neural_net import AbstractNeuralNetwork
from BioHCI.data.data_splitter import DataSplitter
from BioHCI.data_processing.category_balancer import CategoryBalancer
from BioHCI.data_processing.feature_constructor import FeatureConstructor
from BioHCI.definitions.learning_def import LearningDefinition
from BioHCI.definitions.study_parameters import StudyParameters


class CrossValidator(ABC):
    def __init__(self, subject_dict: types.subj_dataset, data_splitter: DataSplitter, feature_constructor:
    FeatureConstructor, category_balancer: CategoryBalancer, neural_net: AbstractNeuralNetwork, parameters:
    StudyParameters, learning_def: LearningDefinition, all_categories: List[str], extra_model_name: str = ""):
        self.__subject_dict = subject_dict
        self.__data_splitter = data_splitter
        self.__feature_constructor = feature_constructor
        self.__category_balancer = category_balancer
        self.__category_map = utils.map_categories(all_categories)
        self.__neural_net = neural_net
        self.__learning_def = learning_def
        self.__parameters = parameters
        self.__num_folds = parameters.num_folds
        self.__extra_model_name = extra_model_name

        self.__all_val_accuracies = []
        self.__all_train_accuracies = []
        self.__all_epoch_train_losses = []

        # declare variables that will contain time needed to compute these operations
        self.__cv_time = ""
        self.__train_time = ""
        self.__val_time = 0

        tbx_name = parameters.study_name + "/tensorboardX_runs"
        self.__tbx_path = utils.create_dir(join(utils.get_root_path("Results"), tbx_name))
        self.__writer = SummaryWriter(self.tbx_path)

        results_log_subdir = self.parameters.study_name + "/learning_logs"
        self.__results_log_path = utils.create_dir(join(utils.get_root_path("Results"), results_log_subdir))
        self.__result_logger = self.define_result_logger()

        model_subdir = parameters.study_name + "/trained_models"
        self.__saved_model_dir = utils.create_dir(join(utils.get_root_path("saved_objects"), model_subdir))

        utils.cleanup(self.model_dir, "_test")

        # create a confusion matrix to track correct guesses (accumulated over all folds of the Cross-Validation
        # below
        # self.__confusion_matrix = torch.zeros(len(all_categories), len(all_categories))

        self.__confusion_matrix = np.zeros((len(all_categories), len(all_categories)))

    def define_result_logger(self) -> logging.Logger:
        """
        Creates a custom logger to write the results of the cross validation on the console and file.

        Returns:
            logger(Logging.logger): the logger used to report statistical results.

        """
        # Create a custom logger
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.DEBUG)

        # Create handlers
        c_handler = logging.StreamHandler()
        c_handler.setLevel(logging.INFO)

        f_handler = logging.FileHandler(filename=self.logfile_path)
        f_handler.setLevel(logging.DEBUG)

        # Add handlers to the logger
        logger.addHandler(c_handler)
        logger.addHandler(f_handler)

        print(f"Logging into file: {self.logfile_path}")
        return logger

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
    def category_map(self):
        return self.__category_map

    @property
    def neural_net(self):
        return self.__neural_net

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
    def general_name(self) -> str:
        name = self.neural_net.name + "-batch-" + str(self.learning_def.batch_size) + "-" + self.extra_model_name
        return name

    @property
    def model_name(self) -> str:
        return self.__model_name

    @model_name.setter
    def model_name(self, model_name: str):
        self.__model_name = model_name

    @property
    def model_path(self) -> str:
        return self.__model_path

    @model_path.setter
    def model_path(self, model_path: str):
        self.__model_path = model_path

    @property
    def model_dir(self) -> str:
        return self.__saved_model_dir

    @property
    def all_val_accuracies(self) -> List[float]:
        return self.__all_val_accuracies

    @property
    def all_train_accuracies(self) -> List[float]:
        return self.__all_train_accuracies

    @property
    def all_epoch_train_losses(self) -> List[float]:
        return self.__all_epoch_train_losses

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
    def result_logger(self) -> logging.Logger:
        return self.__result_logger

    @property
    def confusion_matrix(self):
        return self.__confusion_matrix

    @property
    def extra_model_name(self) -> str:
        return self.__extra_model_name

    def perform_simple_train_val(self) -> None:
        cv_start = time.time()

        feature_dataset = self.feature_constructor.produce_feature_dataset(self.subject_dict)
        train_dataset, val_dataset = self.data_splitter.split_dataset_features(feature_dataset, 0.7)

        # balance each dataset individually
        balanced_train = self.category_balancer.balance(train_dataset)
        balanced_val = self.category_balancer.balance(val_dataset)

        print(f"\nNetwork Architecture: {self.neural_net}\n")

        self.__model_name = self.__produce_model_name()
        self.__model_path = join(self.__saved_model_dir, self.model_name)

        # starting training with the above-defined parameters
        train_start = time.time()
        self.train(balanced_train)
        self.train_time = utils.time_since(train_start)

        # start validating the learning
        val_start = time.time()
        self.val(balanced_val)
        self.val_time = utils.time_since(val_start)

        self.cv_time = utils.time_since(cv_start)
        self.log_cv_results()
        self.draw_confusion_matrix()


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

            print(f"\nNetwork Architecture: {self.neural_net}\n")

            self.__model_name = self.__produce_model_name(i)
            self.__model_path = join(self.__saved_model_dir, self.model_name)

            # starting training with the above-defined parameters
            train_start = time.time()
            self.train(balanced_train)
            self.train_time = utils.time_since(train_start)

            # start validating the learning
            val_start = time.time()
            self.val(balanced_val)
            self.val_time = utils.time_since(val_start)

        self.cv_time = utils.time_since(cv_start)
        self.log_cv_results()
        self.draw_confusion_matrix()

    @abstractmethod
    def _get_data_and_labels(self, python_dataset):
        pass

    @abstractmethod
    def train(self, train_dataset):
        pass

    @abstractmethod
    def val(self, val_dataset):
        pass

    def __produce_model_name(self, i:int=-1) -> str:
        """
        Produces the model name. If cross validation is performed the fold number out of the total ones is
        incorporated into it as well.

        Args:
            i (int): number of cross validation fold (if it exists). If fold number is not included (in case there is no cv), default is -1 (which
                will appear as 0 in the name).

        Returns:

        """
        name = self.general_name + "-fold-" + str(i + 1) + "-" + str(self.num_folds) + ".pt"
        return name

    @property
    def logfile_path(self) -> str:
        name = self.general_name + "_learning_logs.txt"
        results_log_path = join(self.__results_log_path, name)
        return results_log_path

    @property
    def confusion_matrix_path(self) -> str:
        name = self.general_name + "_confusion_matrix.png"
        confusion_path = join(self.__results_log_path, name)
        return confusion_path

    @property
    def avg_train_accuracy(self) -> float:
        """Compute the average of train accuracy of each fold's last epoch."""

        # return the average by dividing the sum by the number of folds (= number of accuracies added)
        avg_accuracy = sum(self.all_train_accuracies) / float(len(self.all_train_accuracies))
        print(f"Average train accuracy over {self.num_folds} folds is  {avg_accuracy:.3f}")
        return avg_accuracy

    @property
    def avg_val_accuracy(self) -> float:
        avg_accuracy = sum(self.all_val_accuracies) / float(len(self.all_val_accuracies))
        print(f"Average validation accuracy over {self.num_folds} folds is {avg_accuracy:.3f}")
        return avg_accuracy

    @property
    def avg_train_losses(self) -> List[float]:
        avg_losses = []
        for i in range(self.learning_def.num_epochs):
            epoch_loss = 0
            for j, loss_list in enumerate(self.all_epoch_train_losses):
                epoch_loss = epoch_loss + loss_list[i]
            avg_losses.append(epoch_loss / self.num_folds)
        return avg_losses

    def get_all_subj_data(self, subj_dict: types.subj_dataset) -> Tuple[List[np.ndarray], List[str]]:
        """
        Creates a dataset of chunks of all subjects with the corresponding categories. At this point the subject data
        is not separated anymore.

        Args:
            subj_dict (dict): a dictionary mapping a subject name to a Subject object

        Returns:
            all_data (ndarray): a 2D numpy array containing the train dataset of shape (instances per sample x number
            of features)
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

    def log_cv_results(self):
        now = datetime.now()
        self.result_logger.info(f"\nTime: {now:%A, %d. %B %Y %I: %M %p}")

        self.result_logger.debug(f"System information: {platform.uname()}")

        self.result_logger.info(f"Dataset Name: {self.parameters.study_name}")
        self.result_logger.info(f"Neural Network: {self.neural_net.name}")

        self.result_logger.info(str(self.neural_net) + "\n")

        self.result_logger.debug(f"Number of original unprocessed attributes: {self.parameters.num_attr}")
        self.result_logger.debug(f"Columns used: {self.parameters.relevant_columns}\n")

        if self.parameters.neural_net:
            self.result_logger.info(f"Was cuda used? - {self.learning_def.use_cuda}")
            self.result_logger.info(
                f"Number of Epochs per cross-validation pass: {self.learning_def.num_epochs}")
            self.result_logger.info(f"Sequence Length: {self.parameters.samples_per_chunk}")
            self.result_logger.info(f"Learning rate: {self.learning_def.learning_rate}")
            self.result_logger.info(f"Batch size: {self.learning_def.batch_size}")
            self.result_logger.info(f"Dropout Rate: {self.learning_def.dropout_rate}\n")

        # some evaluation metrics
        self.result_logger.info(
            f"Training loss of last epoch (avg over cross-validation folds): {self.avg_train_losses[-1]:.2f}")

        # metrics more specific to the cv type
        self._log_specific_results()

        # adding performance information
        self.result_logger.info(f"Performance Metrics:")
        self.result_logger.info(f"Number of threads: {self.parameters.num_threads}")
        self.result_logger.info(
            f"Total cross-validation time ({self.parameters.num_folds} - Fold): {self.cv_time}")
        self.result_logger.info(f"Train time (over last cross-validation pass): {self.train_time}")
        self.result_logger.info(f"Test time (over last cross-validation pass): {self.val_time}")
        self.result_logger.debug(
            "\n***************************************************************************************************\n\n")

        # close and detach all handlers
        handlers = self.result_logger.handlers[:]
        for handler in handlers:
            handler.close()
            self.result_logger.removeHandler(handler)

        logging.shutdown()

    @abstractmethod
    def _log_specific_results(self):
        pass

    def draw_confusion_matrix(self):
        plt.figure(figsize=(14, 10))
        sns.set(font_scale=1.4)
        confusion_matrix_fig = sns.heatmap(self.confusion_matrix, xticklabels=5, yticklabels=5,
                                           cmap=sns.color_palette("RdGy", 10))
        plt.show()

        confusion_matrix_fig.figure.savefig(self.confusion_matrix_path)
        print(f"Saved confusion matrix to {self.confusion_matrix_path}")
