import logging
import os
import pickle
import platform
import time
from abc import ABC, abstractmethod
from datetime import datetime
from os.path import join
from typing import List, Optional, Tuple

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch

import BioHCI.helpers.type_aliases as types
import BioHCI.helpers.utilities as utils
from BioHCI.architectures.abstract_neural_net import AbstractNeuralNetwork
from BioHCI.architectures.cnn2d_lstm_class import CNN2D_LSTM_C
from BioHCI.architectures.cnn_lstm_class import CNN_LSTM_C
from BioHCI.architectures.lstm import LSTM
from BioHCI.architectures.mlp import MLP
from BioHCI.data.data_splitter import DataSplitter
from BioHCI.data_augmentation.vae_generator import VAE_Generator
from BioHCI.data_processing.category_balancer import CategoryBalancer
from BioHCI.data_processing.feature_constructor import FeatureConstructor
from BioHCI.definitions.learning_def import LearningDefinition
from BioHCI.definitions.study_parameters import StudyParameters


class Analyser(ABC):
    def __init__(self, data_splitter: DataSplitter, feature_constructor: FeatureConstructor,
                 category_balancer: CategoryBalancer, parameters: StudyParameters, learning_def: LearningDefinition,
                 all_categories: List[str], extra_model_name: str = ""):
        self.__all_categories = all_categories
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

        results_log_subdir = self.parameters.study_name + "/learning_logs"
        self.__results_log_path = utils.create_dir(join(utils.get_root_path("Results"), results_log_subdir))
        self._result_logger = self.define_result_logger()

        model_subdir = parameters.study_name + "/trained_models"
        self.__saved_model_dir = utils.create_dir(join(utils.get_root_path("saved_objects"), model_subdir))

        confusion_matrix_subdir = parameters.study_name + "/confusion_matrices"
        self.__confusion_matrix_obj_dir = utils.create_dir(join(utils.get_root_path("saved_objects"),
                                                                confusion_matrix_subdir))
        self.__cv_confusion_matrix = np.zeros((len(all_categories), len(all_categories)))
        self.__test_confusion_matrix = np.zeros((len(all_categories), len(all_categories)))
        # utils.cleanup(self.model_dir, "_test")

    @property
    def all_categories(self):
        return self.__all_categories

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
    def result_logger(self) -> logging.Logger:
        return self._result_logger

    @property
    def cv_confusion_matrix(self) -> np.ndarray:
        return self.__cv_confusion_matrix

    @cv_confusion_matrix.setter
    def cv_confusion_matrix(self, cm: np.ndarray):
        self.__cv_confusion_matrix = cm

    @property
    def test_confusion_matrix(self) -> np.ndarray:
        return self.__test_confusion_matrix

    @test_confusion_matrix.setter
    def test_confusion_matrix(self, cm: np.ndarray):
        self.__test_confusion_matrix = cm

    @property
    def extra_model_name(self) -> str:
        return self.__extra_model_name

    @property
    def logfile_path(self) -> str:
        name = self.general_name + "_learning_logs.txt"
        results_log_path = join(self.results_log_dir, name)
        return results_log_path

    @property
    def cv_confusion_matrix_path(self) -> str:
        name = self.general_name + "_cv_confusion_matrix.pdf"
        confusion_path = join(self.results_log_dir, name)
        return confusion_path

    @property
    def cv_confusion_matrix_obj_path(self):
        name = self.general_name + "_cv_confusion_matrix.pt"
        path = join(self.confusion_matrix_obj_dir, name)
        return path

    @property
    def test_confusion_matrix_path(self) -> str:
        name = self.general_name + "_test_confusion_matrix.pdf"
        confusion_path = join(self.results_log_dir, name)
        return confusion_path

    @property
    def test_confusion_matrix_obj_path(self):
        name = self.general_name + "_test_confusion_matrix.pt"
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
        name = self.general_name + extra + "-fold-" + str(i) + "-" + str(self.num_folds) + ".pt"
        return name

    def __init_nn(self) -> AbstractNeuralNetwork:
        if self.learning_def.nn_name == "CNN_LSTM_cl":
            neural_net = CNN_LSTM_C(nn_learning_def=self.learning_def)
        elif self.learning_def.nn_name == "CNN2D_LSTM_cl":
            neural_net = CNN2D_LSTM_C(nn_learning_def=self.learning_def)
        elif self.learning_def.nn_name == "LSTM":
            neural_net = LSTM(nn_learning_def=self.learning_def)
        elif self.learning_def.nn_name == "MLP":
            neural_net = MLP(nn_learning_def=self.learning_def)
        else:
            neural_net = None
            print("Neural Network could not be initialized.")

        assert neural_net is not None
        if self.learning_def.use_cuda:
            neural_net = neural_net.cuda()

        return neural_net

    # @staticmethod
    # def standardize(dataset):
    #     dataset = dataset
    #     means = dataset.mean(dim=1, keepdim=True)
    #     std_dev = dataset.std(dim=1, keepdim=True)
    #     standardized_data = (dataset - means) / std_dev
    #     return standardized_data

    def normalize_all_samples(self, dataset):
        standardized_list = []
        for sample in dataset:
            standardized_sample = self.normalize_sample(sample)
            standardized_list.append(standardized_sample)
        return standardized_list

    @staticmethod
    def normalize_sample(sample):
        means = np.mean(sample, axis=0)
        std_dev = np.std(sample, axis=0)
        standardized_data = (sample - means) / std_dev

        # x = np.arange(0, 250)
        # original = sample[:,2]
        stadardized = standardized_data[:, 2]
        # plt.plot(x, original, label="original")
        # plt.plot(x, stadardized, label="standardized")
        # plt.legend()
        # plt.show()
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
    def perform_cross_validation(self, cv_subject_dict) -> None:
        cv_start = time.time()

        feature_dataset = self.feature_constructor.produce_feature_dataset(cv_subject_dict)
        neural_net = None

        train_fold_losses = []
        train_fold_accuracies = []
        val_fold_losses = []
        val_fold_accuracies = []
        fold_cm = []

        self.log_general_info()
        for i in range(1, self.num_folds + 1):
            self.result_logger.info(
                "\n*************************************************************************************************")
            self.result_logger.info(f"Cross-Validation Fold: {i} / {self.num_folds}\n")

            # in data splitter, val_index starts from 0, while i in this case starts from 1
            train_dataset, val_dataset = self.data_splitter.split_into_folds_features(
                feature_dictionary=feature_dataset, num_folds=self.num_folds, val_index=(i - 1))

            self.result_logger.info(f"\nSubjects in training dataset: ")
            self.print_dataset_subj(train_dataset)
            self.result_logger.info(f"Subjects in val dataset: ")
            self.print_dataset_subj(val_dataset)

            # train_dataset, val_dataset = self.data_splitter.split_into_folds_raw(
            #     subject_dictionary=self.subject_dict, num_folds=self.num_folds, val_index=i)

            # train_feature = self.feature_constructor.produce_feature_dataset(train_dataset)
            # val_feature = self.feature_constructor.produce_feature_dataset(val_dataset)

            # balance each dataset individually
            balanced_train = self.category_balancer.balance(train_dataset)
            balanced_val = self.category_balancer.balance(val_dataset)

            if self.parameters.neural_net:
                neural_net = self.__init_nn()
                if i == 1:
                    self.result_logger.info(f"Neural Network: {str(neural_net)}\n")
            # the stochastic gradient descent function to update weights and biases
            optimizer = torch.optim.Adam(neural_net.parameters(), lr=self.learning_def.learning_rate)

            self.__model_name = self._produce_model_name(i=i)
            self.__model_path = join(self.__saved_model_dir, self.model_name)
            self.result_logger.info(f"\nModel path: {self.model_path}\n")

            current_cm = np.zeros((len(self.all_categories), len(self.all_categories)))
            # starting training and evaluation with the above-defined parameters
            train_loss, train_accuracy, val_loss, val_accuracy = \
                self._specific_train_val(balanced_train, balanced_val, neural_net, optimizer, current_cm, i)
            train_fold_losses.append(train_loss)
            train_fold_accuracies.append(train_accuracy)
            val_fold_losses.append(val_loss)
            val_fold_accuracies.append(val_accuracy)

            # save current value of confusion matrix in a list, and zero it out for next fold
            fold_cm.append(current_cm)

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

        # compute confusion matrix as the average of confusion matrices of all folds
        self.cv_confusion_matrix = sum(fold_cm) / len(fold_cm)
        print(f"\nFinal cross-validatoin confusion matrix:\n {self.cv_confusion_matrix}")

        self.save_confusion_matrix(self.cv_confusion_matrix, self.cv_confusion_matrix_obj_path,
                                   self.cv_confusion_matrix_path)
        self.compute_cm_stats(self.cv_confusion_matrix)

    def evaluate_all_models(self, test_subject_dict, model_paths: Optional[List] = None):
        self.result_logger.debug(
            "\n********************************************************************************************************"
            "\n********************************************************************************************************"
            "\n")
        self.result_logger.info(f"\nEvaluation Only Results!!!!!!")
        self.log_general_info()

        fold_cm = []
        fold_accuracies = []
        for i in range(1, self.num_folds + 1):
            current_cm = np.zeros((len(self.all_categories), len(self.all_categories)))
            model_name = self._produce_model_name(i=i)
            if model_paths is None:
                model_path = join(self.__saved_model_dir, model_name)
            else:
                model_path = model_paths[i]

            self.result_logger.info(
                "\n***********************************************************************************************")
            self.result_logger.info(f"Run: {i}")

            val_accuracy = self.eval_only(test_subject_dict, confusion_matrix=current_cm, model_path=model_path)
            fold_accuracies.append(val_accuracy)
            fold_cm.append(current_cm)

        self.test_confusion_matrix = sum(fold_cm) / len(fold_cm)

        avg_acc = sum(fold_accuracies) / len(fold_accuracies)
        self.result_logger.info(f"\nAverage test accuracy is: {avg_acc:.3f}")
        self.result_logger.info(f"\nFinal test confusion matrix:\n {self.test_confusion_matrix}")

        self.save_confusion_matrix(self.test_confusion_matrix, self.test_confusion_matrix_obj_path,
                                   self.test_confusion_matrix_path)
        self.compute_cm_stats(self.test_confusion_matrix)
        self.result_logger.info(
            "\n***************************************************************************************************\n\n")

    def print_dataset_subj(self, dataset):
        for subj_name, subj in dataset.items():
            self.result_logger.info(subj_name)
        self.result_logger.info("\n")

    def train_only(self, train_subject_dict, neural_net):
        feature_dataset = self.feature_constructor.produce_feature_dataset(train_subject_dict)
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

    def eval_only(self, test_subject_dict, confusion_matrix, model_path=None):

        feature_dataset = self.feature_constructor.produce_feature_dataset(test_subject_dict)
        balanced_val = self.category_balancer.balance(feature_dataset)

        self.result_logger.info(f"\nVal dataset: ")
        self.print_dataset_subj(balanced_val)

        self.result_logger.info(f"Model from: {model_path}")
        val_accuracy = self._specific_eval_only(balanced_val, confusion_matrix, model_path=model_path)

        return val_accuracy

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

    def save_confusion_matrix(self, matrix: np.ndarray, cm_path: str, fig_path: str):
        with open(cm_path, 'wb') as f:
            pickle.dump(matrix, f, pickle.HIGHEST_PROTOCOL)
        print(f"Saved confusion matrix object (.pkl) to : {cm_path}")

        # draw and save figure
        self.draw_confusion_matrix(matrix, fig_path, labels=True)

    def compute_cm_stats(self, confusion_matrix: np.ndarray):

        # https: // stackoverflow.com / a / 48101802

        self.result_logger.info(
            "\n***************************************************************************************************\n\n")
        tp = np.diag(confusion_matrix)  # true positives
        self.result_logger.info(f"TP:\n {tp}\n")
        fp = np.sum(confusion_matrix, axis=0) - tp  # false positives
        self.result_logger.info(f"FP:\n {fp}\n")
        fn = np.sum(confusion_matrix, axis=1) - tp  # false negatives
        self.result_logger.info(f"FN:\n {fn}\n")

        num_classes = confusion_matrix.shape[0]
        tn = []
        for i in range(num_classes):
            temp = np.delete(confusion_matrix, i, 0)  # delete ith row
            temp = np.delete(temp, i, 1)  # delete ith column
            tn.append(sum(sum(temp)))
        self.result_logger.info(f"TN:\n {tn}\n")

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

    def generate_confusion_matrix_fig_from_obj_name(self, cm_name: str, fig_path: str) -> None:
        """
        Given a confusion matrix name, produces its confusion matrix figure.

        Args:
            cm_name: the name of the pickled confusion matrix object to convert into a figure
            fig_path: path to save figure
        """
        # assert cm_name.endswith(".pkl")
        path = join(self.confusion_matrix_obj_dir, cm_name)
        if os.path.exists(path):
            with (open(path, "rb")) as openfile:
                confusion_matrix = pickle.load(openfile)
                self.draw_confusion_matrix(confusion_matrix, fig_path)

    def draw_confusion_matrix(self, confusion_matrix: np.ndarray, fig_path: str, labels: bool = False):
        plt.figure(figsize=(55, 40))
        cmap = "magma_r"
        sns.set(font_scale=6)

        # if labels is set to true, get the categories from the internal category map, and convert confusion matrix to
        # a dataframe to be able to plot with them.
        if labels is True:
            cat_list = sorted(self.category_map.keys())
            cat_list[-1] = '?'
            confusion_matrix = pd.DataFrame(confusion_matrix, columns=cat_list, index=cat_list)

        confusion_matrix_fig = sns.heatmap(confusion_matrix, xticklabels=1, yticklabels=1, cmap=cmap)
        plt.show()

        confusion_matrix_fig.figure.savefig(fig_path, dpi=500)
        self.result_logger.info(f"\nSaved confusion matrix figure (.png) to {fig_path}")

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

    def _specific_train_val(self, balanced_train, balanced_val, neural_net, optimizer, current_cm, i):
        return None, None, None, None

    @abstractmethod
    def _specific_train_only(self, balanced_train, neural_net, optimizer):
        pass

    def _specific_eval_only(self, balanced_val, confusion_matrix, model_path=None):
        """

        Returns:
            val_accuracy: accuracy of validation
        """
        pass

    @abstractmethod
    def _get_data_and_labels(self, python_dataset):
        pass

    @abstractmethod
    def train(self, train_dataset, neural_net, optimizer):
        pass

    @abstractmethod
    def val(self, val_dataset, confusion_matrix):
        pass
