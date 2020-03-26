import logging
import os
import pickle
import platform
import time
from abc import ABC, abstractmethod
from datetime import datetime
from os.path import join
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from tensorboardX import SummaryWriter

import BioHCI.helpers.type_aliases as types
import BioHCI.helpers.utilities as utils
from BioHCI.architectures.abstract_neural_net import AbstractNeuralNetwork
from BioHCI.architectures.cnn2d_lstm_class import CNN2D_LSTM_C
from BioHCI.architectures.cnn_lstm_class import CNN_LSTM_C
from BioHCI.architectures.lstm import LSTM
from BioHCI.data.data_splitter import DataSplitter
from BioHCI.data_processing.category_balancer import CategoryBalancer
from BioHCI.data_processing.feature_constructor import FeatureConstructor
from BioHCI.definitions.learning_def import LearningDefinition
from BioHCI.definitions.study_parameters import StudyParameters


class CrossValidator(ABC):
    def __init__(self, subject_dict: types.subj_dataset, data_splitter: DataSplitter,
                 feature_constructor: FeatureConstructor, category_balancer: CategoryBalancer,
                 parameters: StudyParameters, learning_def: LearningDefinition, all_categories: List[str],
                 extra_model_name: str = ""):
        self.__subject_dict = subject_dict
        self.__data_splitter = data_splitter
        self.__feature_constructor = feature_constructor
        self.__category_balancer = category_balancer
        self.__category_map = utils.map_categories(all_categories)
        self.__learning_def = learning_def
        self.__parameters = parameters
        self.__num_folds = parameters.num_folds
        self.__extra_model_name = extra_model_name
        self.__classification = parameters.classification

        self.__model_name = ""
        self.__model_path = ""

        tbx_name = parameters.study_name + "/tensorboardX_runs"
        self.__tbx_path = utils.create_dir(join(utils.get_root_path("Results"), tbx_name))
        self.__writer = SummaryWriter(self.tbx_path)

        results_log_subdir = self.parameters.study_name + "/learning_logs"
        self.__results_log_path = utils.create_dir(join(utils.get_root_path("Results"), results_log_subdir))
        self._result_logger = self.define_result_logger()

        model_subdir = parameters.study_name + "/trained_models"
        self.__saved_model_dir = utils.create_dir(join(utils.get_root_path("saved_objects"), model_subdir))

        confusion_matrix_subdir = parameters.study_name + "/confusion_matrices"
        self.__confusion_matrix_obj_dir = utils.create_dir(join(utils.get_root_path("saved_objects"),
                                                                confusion_matrix_subdir))
        # utils.cleanup(self.model_dir, "_test")

        # create a confusion matrix to track correct guesses (accumulated over all folds of the Cross-Validation
        # below
        # self.__confusion_matrix = torch.zeros(len(all_categories), len(all_categories))

        # self.__confusion_matrix = np.zeros((len(all_categories), len(all_categories)))
        self.__confusion_matrix = np.zeros((36, 36))

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
        name = self.learning_def.nn_name + "-batch-" + str(self.learning_def.batch_size) + "-" + self.extra_model_name
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
    def results_log_dir(self) -> str:
        return self.__results_log_path

    @property
    def confusion_matrix_obj_dir(self) -> str:
        return self.__confusion_matrix_obj_dir

    @property
    def tbx_path(self):
        return self.__tbx_path

    @property
    def writer(self):
        return self.__writer

    @property
    def result_logger(self) -> logging.Logger:
        return self._result_logger

    @property
    def confusion_matrix(self):
        return self.__confusion_matrix

    @property
    def extra_model_name(self) -> str:
        return self.__extra_model_name

    @property
    def logfile_path(self) -> str:
        name = self.general_name + "_learning_logs.txt"
        results_log_path = join(self.results_log_dir, name)
        return results_log_path

    @property
    def confusion_matrix_path(self) -> str:
        name = self.general_name + "_confusion_matrix.pdf"
        confusion_path = join(self.results_log_dir, name)
        return confusion_path

    @property
    def confusion_matrix_obj_path(self):
        name = self.general_name + "_confusion_matrix.pt"
        path = join(self.confusion_matrix_obj_dir, name)
        return path

    def define_result_logger(self) -> logging.Logger:
        """
        Creates a custom logger to write the results of the cross validation on the console and file.
        Returns:
            logger(Logging.logger): the logger used to report statistical results.
        """
        # Create a custom logger
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.DEBUG)
        logger.propagate = False  # events logged to this logger will not be passed to the handlers of higher level (
        # ancestor) loggers

        # Create handlers
        c_handler = logging.StreamHandler()
        c_handler.setLevel(logging.INFO)
        # set a format which is simpler for console use
        formatter = logging.Formatter('%(message)s')
        # tell the handler to use this format
        c_handler.setFormatter(formatter)

        f_handler = logging.FileHandler(filename=self.logfile_path)
        f_handler.setLevel(logging.DEBUG)

        if logger.hasHandlers():
            logger.handlers.clear()
        # Add handlers to the logger
        logger.addHandler(c_handler)
        logger.addHandler(f_handler)

        print(f"Logging into file: {self.logfile_path}")
        return logger

    def _produce_model_name(self, extra: str = "", i: int = -1) -> str:
        """
        Produces the model name. If cross validation is performed the fold number out of the total ones is
        incorporated into it as well.

        Args:
            i (int): number of cross validation fold (if it exists). If fold number is not included (in case there is
            no cv), default is -1 (which will appear as 0 in the name).

        Returns:

        """
        name = self.general_name + extra + "-fold-" + str(i + 1) + "-" + str(self.num_folds) + ".pt"
        return name

    def __init_nn(self) -> AbstractNeuralNetwork:
        if self.learning_def.nn_name == "CNN_LSTM_cl":
            neural_net = CNN_LSTM_C(nn_learning_def=self.learning_def)
        elif self.learning_def.nn_name == "CNN2D_LSTM_cl":
            neural_net = CNN2D_LSTM_C(nn_learning_def=self.learning_def)
        elif self.learning_def.nn_name == "LSTM":
            neural_net = LSTM(nn_learning_def=self.learning_def)
        else:
            neural_net = None
            print("Neural Network could not be initialized.")

        assert neural_net is not None
        if self.learning_def.use_cuda:
            neural_net = neural_net.cuda()

        return neural_net

    @staticmethod
    def standardize(dataset):
        dataset = dataset
        means = dataset.mean(dim=1, keepdim=True)
        std_dev = dataset.std(dim=1, keepdim=True)
        standardized_data = (dataset - means) / std_dev
        return standardized_data

    @staticmethod
    def get_all_subj_data(subj_dict: types.subj_dataset, seq: bool = True) -> Tuple[List[np.ndarray], List[str]]:
        """
        Creates a dataset of chunks of all subjects with the corresponding categories. At this point the subject data
        is not separated anymore.

        Args:
            subj_dict (dict): a dictionary mapping a subject name to a Subject object
            seq (bool): indicates whether we are using a sequential network (like LSTM)

        Returns:
            all_data (ndarray): a 2D numpy array containing the train dataset of shape (instances per sample x number
            of features)
            all_cat (ndarray): a 1D numpy array containing the category labels of all_data, of shape (number of
                chunks).
        """

        # data to stack - subjects end up mixed together in the ultimate dataset
        all_data = []
        # list of all categories to return
        all_cat = []

        for subj_name, subj in subj_dict.items():
            for i, data in enumerate(subj.data):
                if not seq:
                    data = data.flatten()
                all_data.append(data.astype(np.float32))
                all_cat.append(subj.categories[i])

        return all_data, all_cat

    # TODO: fix how msd dataset is named if we are doing only training (no cv folds)
    def perform_cross_validation(self) -> None:
        cv_start = time.time()

        feature_dataset = self.feature_constructor.produce_feature_dataset(self.subject_dict)
        neural_net = None

        train_fold_losses = []
        train_fold_accuracies = []
        val_fold_losses = []
        val_fold_accuracies = []

        self.log_general_info()
        for i in range(0, self.num_folds):
            self.result_logger.info(
                "\n*************************************************************************************************")
            self.result_logger.info(f"Run: {i}\n")

            train_dataset, val_dataset = self.data_splitter.split_into_folds_features(
                feature_dictionary=feature_dataset, num_folds=self.num_folds, val_index=i)

            # train_dataset, val_dataset = self.data_splitter.split_into_folds_raw(
            #     subject_dictionary=self.subject_dict, num_folds=self.num_folds, val_index=i)

            # train_feature = self.feature_constructor.produce_feature_dataset(train_dataset)
            # val_feature = self.feature_constructor.produce_feature_dataset(val_dataset)

            # balance each dataset individually
            balanced_train = self.category_balancer.balance(train_dataset)
            balanced_val = self.category_balancer.balance(val_dataset)

            if self.parameters.neural_net:
                neural_net = self.__init_nn()
                if i == 0:
                    self.result_logger.info(f"Neural Network: {str(neural_net)}\n")
            # the stochastic gradient descent function to update weights a self.perform_cross_validation()nd biases
            optimizer = torch.optim.Adam(neural_net.parameters(), lr=self.learning_def.learning_rate)

            self.__model_name = self._produce_model_name(i=i)
            self.__model_path = join(self.__saved_model_dir, self.model_name)

            # starting training and evaluation with the above-defined parameters
            train_loss, train_accuracy, val_loss, val_accuracy = \
                self._specific_train_val(balanced_train, balanced_val, neural_net, optimizer, i)
            train_fold_losses.append(train_loss)
            train_fold_accuracies.append(train_accuracy)
            val_fold_losses.append(val_loss)
            val_fold_accuracies.append(val_accuracy)

        cv_time = utils.time_since(cv_start)

        # log cross-validation cummulative results
        self.result_logger.info(
            "\n---------------------------------------------------------------------------------------------------\n\n")
        self.result_logger.info(
            f"Total cross-validation time ({self.parameters.num_folds} - Fold): {cv_time}")
        self.result_logger.info(
            f"Train Losses over all folds: {self.format_list(train_fold_losses)}")
        self.result_logger.info(
            f"Train Accuracies over all folds: {self.format_list(train_fold_accuracies)}")
        self.result_logger.info(
            f"Validation Losses over all folds: {self.format_list(val_fold_losses)}")
        self.result_logger.info(
            f"Validation Accuracies over all folds: {self.format_list(val_fold_accuracies)}")

        avg_train_acc = sum(train_fold_accuracies) / len(train_fold_accuracies)
        avg_val_acc = sum(val_fold_accuracies) / len(val_fold_accuracies)
        self.result_logger.info(
            f"\nAverage Train Accuracy: {avg_train_acc : .3f}")
        self.result_logger.info(
            f"Average Val Accuracy: {avg_val_acc : .3f}")

        self.save_confusion_matrix(self.confusion_matrix)
        self.compute_cm_stats(self.confusion_matrix)
        # self.close_logger()

    def train_only(self, neural_net):
        feature_dataset = self.feature_constructor.produce_feature_dataset(self.subject_dict)
        balanced_train = self.category_balancer.balance(feature_dataset)

        self.__model_name = self.general_name + ".pt"
        self.__model_path = join(self.__saved_model_dir, self.model_name)

        self.log_general_info()
        self.result_logger.info(f"\nTrain Only Results!!!!!!")
        self.result_logger.info(f"\nNetwork Architecture: {neural_net}\n")

        # the stochastic gradient descent function to update weights a self.perform_cross_validation()nd biases
        optimizer = torch.optim.Adam(neural_net.parameters(), lr=self.learning_def.learning_rate)

        # train
        self._specific_train_only(balanced_train, neural_net, optimizer)
        torch.save(neural_net, self.model_path)

        self.result_logger.info(f"Saved model to {self.model_path}")
        self.result_logger.debug(
            "\n***************************************************************************************************\n\n")
        self.close_logger()

    def eval_only(self, model_path=None):

        feature_dataset = self.feature_constructor.produce_feature_dataset(self.subject_dict)
        balanced_val = self.category_balancer.balance(feature_dataset)

        self.log_general_info()
        self.result_logger.info(f"\nEvaluation Only Results!!!!!!")

        self.result_logger.info(f"Model from: {model_path}")
        self._specific_eval_only(balanced_val, model_path=model_path)

        self.save_confusion_matrix(self.confusion_matrix)
        self.compute_cm_stats(self.confusion_matrix)
        self.result_logger.debug(
            "\n***************************************************************************************************\n\n")
        self.close_logger()

    def log_general_info(self):
        self.result_logger.debug(
            "\n********************************************************************************************************"
            "\n********************************************************************************************************"
            "\n\n")

        now = datetime.now()
        self.result_logger.info(f"\nTime: {now:%A, %d. %B %Y %I: %M %p}")

        self.result_logger.debug(f"System information: {platform.uname()}")

        self.result_logger.info(f"Dataset Name: {self.parameters.study_name}")

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
        self.result_logger.info(f"Number of threads: {self.parameters.num_threads}\n")

    def save_confusion_matrix(self, matrix):
        with open(self.confusion_matrix_obj_path, 'wb') as f:
            pickle.dump(matrix, f, pickle.HIGHEST_PROTOCOL)
        print(f"Saved confusion matrix object (.pkl) to : {self.confusion_matrix_obj_path}")

        # draw and save figure
        self.draw_confusion_matrix(matrix)

    def compute_cm_stats(self, confusion_matrix: np.ndarray):

        # https: // stackoverflow.com / a / 48101802

        self.result_logger.info(
            "\n***************************************************************************************************\n\n")
        tp = np.diag(confusion_matrix)  # true positives
        self.result_logger.info(f"TP: {tp}")
        fp = np.sum(confusion_matrix, axis=0) - tp  # false positives
        self.result_logger.info(f"FP: {fp}")
        fn = np.sum(confusion_matrix, axis=1) - tp  # false negatives
        self.result_logger.info(f"FN: {fn}")

        num_classes = confusion_matrix.shape[0]
        tn = []
        for i in range(num_classes):
            temp = np.delete(confusion_matrix, i, 0)  # delete ith row
            temp = np.delete(temp, i, 1)  # delete ith column
            tn.append(sum(sum(temp)))
        self.result_logger.info(f"TN: {tn}\n")

        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1_score = 2 * (precision * recall) / (precision + recall)

        self.result_logger.info(f"Precision: {self.format_list(precision)}")
        self.result_logger.info(f"Recall: {self.format_list(recall)}")
        self.result_logger.info(f"F1 Score: {self.format_list(f1_score)}\n")

        self.result_logger.info(f"Macro Precision: {sum(precision) / len(precision): .3f}")
        self.result_logger.info(f"Macro Recall: {sum(recall) / len(precision): .3f}")
        self.result_logger.info(f"Macro F1 Score: {sum(f1_score) / len(f1_score): .3f}\n")
        return

    def generate_confusion_matrix_fig_from_obj_name(self, cm_name: str) -> None:
        """
        Given a confusion matrix name, produces its confusion matrix figure.

        Args:
            cm_name: the name of the pickled confusion matrix object to convert into a figure
        """
        # assert cm_name.endswith(".pkl")
        path = join(self.confusion_matrix_obj_dir, cm_name)
        if os.path.exists(path):
            with (open(path, "rb")) as openfile:
                confusion_matrix = pickle.load(openfile)
                self.draw_confusion_matrix(confusion_matrix)

    def draw_confusion_matrix(self, confusion_matrix: np.ndarray):
        plt.figure(figsize=(55, 40))
        sns.set(font_scale=4)
        confusion_matrix_fig = sns.heatmap(confusion_matrix, xticklabels=np.arange(1, 37), yticklabels=np.arange(
            1, 37), cmap="YlGnBu")
        # confusion_matrix_fig = sns.heatmap(confusion_matrix, xticklabels=1, yticklabels=1, cmap="YlGnBu")
        # cmap = sns.color_palette("Reds", 10))
        # cmap = "YlGnBu"))
        plt.show()

        confusion_matrix_fig.figure.savefig(self.confusion_matrix_path, dpi=500)
        self.result_logger.info(f"\nSaved confusion matrix figure (.png) to {self.confusion_matrix_path}")

    def close_logger(self):
        self.result_logger.debug(
            "\n********************************************************************************************************"
            "\n********************************************************************************************************"
            "\n\n")

        # close and detach all handlers
        handlers = self.result_logger.handlers[:]
        for handler in handlers:
            handler.close()
            self.result_logger.removeHandler(handler)

        logging.shutdown()

    @staticmethod
    def format_list(float_ls: List[Optional[float]]) -> List[str]:
        my_formatted_list = ['%.3f' % elem for elem in float_ls]
        return my_formatted_list

    def _specific_train_val(self, balanced_train, balanced_val, neural_net, optimizer):
        return None, None, None, None

    @abstractmethod
    def _specific_train_only(self, balanced_train, neural_net, optimizer):
        pass

    def _specific_eval_only(self, balanced_val, model_path=None):
        pass

    @abstractmethod
    def _get_data_and_labels(self, python_dataset):
        pass

    @abstractmethod
    def train(self, train_dataset, neural_net, optimizer):
        pass

    @abstractmethod
    def val(self, val_dataset):
        pass
