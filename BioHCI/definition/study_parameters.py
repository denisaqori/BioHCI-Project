"""
© Denisa Qori McDonald 2018 All Rights Reserved
"""

import os
import re

from BioHCI.helpers import utilities as util
from typing import List, Optional, Match, Iterator


class StudyParameters(object):
    def __init__(self, attr_dict=None):
        # directory where the text data to be processed is found (all files in there are used)
        # 'Resources/fNIRS_boredom_data' correspond to boredom data
        self.__dir_path = None

        # the name of the study
        self.__study_name = None
        # the type of sensor data used - currently not being used anywhere but for bookkeeping
        self.__sensor_name = None
        self.__file_format = None

        # data information

        # these values represent column indices to retain from the data in each file
        # self.__relevant_columns = [1, 5, 9, 13, 17, 21]
        self.__relevant_columns = None
        self.__column_names = None
        # this value represents the beginning of valid rows in the file
        self.__start_row = None

        # how arrays of data will be named: so far, either from subdirectories of each subject, or from filenames of
        # each subject
        self.__cat_names = None

        # the total number of subjects
        self.__num_subj = None
        # the column in which the file for each subject contains labels. If set to None, there are no labels.
        self.__labels_col = None
        self.__plot_labels = None
        # data processing information

        # whether the dataset is to be standardized (default is per subject per column)
        self.__standardize = None

        # whether to convert to frequency domain
        self.__compute_fft = None
        # sampling frequency
        self.__sampling_freq = None
        # number of samples in windowing segment
        self.__nfft = None

        self.__chunk_instances = None
        self.__samples_per_chunk = None  # when chunking the dataset, the number of instances/samples in one chunk
        self.__interval_overlap = None  # determining whether to overlap instances while chunking

        self.__construct_features = None  # determining whether to construct features
        self.__feature_window = None  # the number of measurements over which to define features (collapsing all to 1)
        self.__feature_overlap = None  # determining whether to overlap instances while constructing features

        self.__num_folds = None  # The number of folds for cross-validation

        # run information
        self.__num_threads = None  # The number of threads to be used during training (for gradient computing and
        # loading)
        self.__neural_net = None

        if attr_dict is not None:
            for attribute, value in attr_dict.items():
                if value == "None":
                    value = None
                setattr(self, attribute, value)

    def __str__(self) -> str:
        s = "\nStudyParameters: \n"
        s = s + "\n**********************************************************"
        s = s + "\nData source directory: " + str(self.dir_path)
        s = s + "\nStudy name: " + str(self.study_name)
        s = s + "\nSensor type: " + str(self.sensor_name)
        s = s + "\nFile format to process: " + str(self.file_format)
        s = s + "\nColumns to keep from each file: " + str(self.relevant_columns)
        s = s + "\nColumns names: " + str(self.column_names)
        s = s + "\nInitial row index: " + str(self.start_row)
        s = s + "\nNumber of subjects: " + str(self.num_subj)
        s = s + "\nLabels column index: " + str(self.labels_col)
        s = s + "\nPlot Labels: " + str(self.plot_labels)
        s = s + "\nShould the data be standardized?: " + str(self.standardize)
        s = s + "\nShould the data be converted to frequency domain?: " + str(self.compute_fft)
        s = s + "\nNumber of samples in windowing segment (for freq): " + str(self.nfft)
        s = s + "\nSampling frequency: " + str(self.sampling_freq)
        s = s + "\nNumber of samples per chunk/window: " + str(self.samples_per_chunk)
        s = s + "\nShould we create a chunk by overlapping previous and next chunks (half of each)?: " + str(
            self.interval_overlap)
        s = s + "\nShould we create features over these chunks?: " + str(self.construct_features)
        s = s + "\nIf yes, over what interval (number of inst.)? If not, answer should be None: " + str(
            self.feature_window)
        s = s + "\nShould we create a feature intervals by overlapping previous and next intervals (half of each)?: " \
            + str(self.feature_overlap)
        s = s + "\nNumber of cross-validation folds: " + str(self.num_folds)
        s = s + "\nNumber of threads: " + str(self.num_threads)
        s = s + "\nAre we using neural networks?: " + str(self.neural_net)
        s = s + "\n**********************************************************\n"
        return s

    @property
    def dir_path(self) -> Optional[str]:
        return self.__dir_path

    @dir_path.setter
    def dir_path(self, dir_path: str) -> None:
        project_root_path = util.get_root_path("Resources")
        path = os.path.abspath(os.path.join(project_root_path, dir_path))

        assert (os.path.exists(path)), "The directory \'" + path + "\' does not exist. Ensure the dataset is " \
                                                                   "properly placed."
        self.__dir_path = path

    @property
    def study_name(self) -> Optional[str]:
        return self.__study_name

    @study_name.setter
    def study_name(self, study_name: str) -> None:
        self.__study_name = study_name

    @property
    def sensor_name(self) -> Optional[str]:
        return self.__sensor_name

    @sensor_name.setter
    def sensor_name(self, sensor_name: str) -> None:
        self.__sensor_name = sensor_name

    @property
    def file_format(self) -> Optional[str]:
        return self.__file_format

    @file_format.setter
    def file_format(self, file_format: str) -> None:
        assert file_format.startswith('.'), "The file ending name should start with a '.'"
        self.__file_format = file_format

    @property
    def relevant_columns(self) -> Optional[List[int]]:
        return self.__relevant_columns

    @relevant_columns.setter
    def relevant_columns(self, relevant_columns) -> None:
        self.__relevant_columns: List[int] = []

        if isinstance(relevant_columns, str):
            structure = re.compile(r'\[\d*:\d+\]')
            assert structure.match(
                relevant_columns), "The structure of the slice is not proper. A colon needs to be included in the " \
                                   "string with the start and end index on each side. If the first index is " \
                                   "omitted, 0 is assumed. All these elements need to be within square brackets."
            colon = re.search(r':', relevant_columns)
            col_start = colon.start()

            start_idx = 0
            end_idx = 0

            numbers: Iterator[Match[str]] = re.finditer(r'\d+', relevant_columns)
            for number in numbers:
                if number.start() < col_start:
                    start_idx = int(number.group())
                if number.start() > col_start:
                    end_idx = int(number.group())

            self.__relevant_columns = list(range(start_idx, end_idx))

        else:
            assert (isinstance(relevant_columns, list)), "The relevant columns to process need to be passed in a list."
            for elem in relevant_columns:
                assert (isinstance(elem, int) and int(elem) >= 0), "Each element of the list needs to be a positive " \
                                                               "integer or zero."
                self.__relevant_columns.append(int(elem))

    @property
    def column_names(self) -> Optional[List[str]]:
        return self.__column_names

    @column_names.setter
    def column_names(self, column_names: List[str]):
        if column_names is not None:
            assert len(column_names) == len(self.__relevant_columns), "There needs to be a column name for each relevant " \
                                                                  "column."
        self.__column_names = column_names

    @property
    def start_row(self) -> Optional[int]:
        return self.__start_row

    @start_row.setter
    def start_row(self, start_row: int) -> None:
        assert int(start_row) >= 0, "Number of subjects needs to be a positive integer or zero."
        self.__start_row = start_row

    @property
    def cat_names(self) -> Optional[str]:
        return self.__cat_names

    @cat_names.setter
    def cat_names(self, cat_names: str) -> None:
        assert cat_names == 'file'.lower() or cat_names == 'dir'.lower()
        self.__cat_names = cat_names.lower()

    @property
    def num_subj(self) -> Optional[int]:
        return self.__num_subj

    @num_subj.setter
    def num_subj(self, num_subj: int) -> None:
        assert int(num_subj) > 0, "Number of subjects needs to be a positive integer."
        self.__num_subj = num_subj

    @property
    def num_attr(self) -> Optional[int]:
        return len(self.__relevant_columns)

    @property
    def labels_col(self) -> Optional[int]:
        return self.__labels_col

    @labels_col.setter
    def labels_col(self, labels_col) -> None:
        if labels_col is "None":
            self.__labels_col = None
        else:
            assert (isinstance(labels_col, int) and int(labels_col)) >= 0, "Label column should be 0 or a positive " \
                                                                           "integer if labels are included with the " \
                                                                           "data, and \"None\" otherwise."
            self.__labels_col = labels_col

    @property
    def plot_labels(self) -> Optional[List[str]]:
        return self.__plot_labels

    @plot_labels.setter
    def plot_labels(self, plot_labels: List[str]) -> None:
        assert (isinstance(plot_labels, list)), "The plot labels to process need to be passed in a list."
        for elem in plot_labels:
            assert isinstance(elem, str), "Each element of the list needs to be a string."
        self.__plot_labels = plot_labels

    @property
    def standardize(self) -> Optional[bool]:
        return self.__standardize

    @standardize.setter
    def standardize(self, standardize: bool) -> None:
        assert (isinstance(standardize, bool)), "The standardize variable needs to be a boolean to indicate " \
                                                "whether the dataset is to be standardized."
        self.__standardize = standardize

    @property
    def compute_fft(self) -> Optional[bool]:
        return self.__compute_fft

    @compute_fft.setter
    def compute_fft(self, compute_fft: bool):
        self.__compute_fft = compute_fft

    @property
    def nfft(self) -> Optional[int]:
        return self.__nfft

    @nfft.setter
    def nfft(self, nfft) -> None:
        if self.compute_fft is True:
            assert (isinstance(nfft, int) and int(nfft) > 0), "FFT windowing segment needs to be a positive integer."
            self.__nfft = nfft
        else:
            assert (self.nfft is None), "If the compute_fft attribute is set to False, " \
                                                 "the nfft attribute needs to be set to \"None\"."
            self.__nfft = None

    @property
    def sampling_freq(self) -> Optional[int]:
        return self.__sampling_freq

    @sampling_freq.setter
    def sampling_freq(self, sampling_freq) -> None:
        if self.compute_fft is True:
            assert (isinstance(sampling_freq, int) and int(sampling_freq)) > 0, "Sampling frequency needs to be a " \
                                                                            "positive integer."
            self.__sampling_freq = sampling_freq
        else:
            assert (self.sampling_freq is None), "If the compute_fft attribute is set to False, " \
                                                 "the sampling_freq attribute needs to be set to \"None\"."
            self.__sampling_freq = None

    @property
    def num_threads(self) -> Optional[int]:
        return self.__num_threads

    @num_threads.setter
    def num_threads(self, num_threads: int) -> None:
        assert num_threads > 0, "Number of threads needs to be a positive integer."
        self.__num_threads = num_threads

    @property
    def neural_net(self) -> Optional[bool]:
        return self.__neural_net

    @neural_net.setter
    def neural_net(self, neural_net: bool) -> None:
        self.__neural_net = neural_net

    @property
    def construct_features(self) -> Optional[bool]:
        return self.__construct_features

    @construct_features.setter
    def construct_features(self, construct_features: bool) -> None:
        self.__construct_features = construct_features

    @property
    def feature_window(self) -> Optional[int]:
        return self.__feature_window

    @feature_window.setter
    def feature_window(self, feature_window) -> None:
        if self.construct_features is True:
            assert (isinstance(feature_window, int) and (int(feature_window) > 0)), "Feature window needs to be a " \
                                                                                    "positive integer."
            self.__feature_window = feature_window
        else:
            assert (feature_window is "None"), "If the construct_features attribute is set to False, " \
                                               "the feature_window attribute needs to be set to \"None\"."
            self.__feature_window = None

    @property
    def feature_overlap(self) -> Optional[bool]:
        return self.__feature_overlap

    @feature_overlap.setter
    def feature_overlap(self, feature_overlap: bool) -> None:
        assert self.construct_features is True, "In order for feature_overlap to be set, construct_features needs to " \
                                                "be True."
        assert isinstance(feature_overlap, bool), "The variable feature_overlap needs to be a boolean."
        self.__feature_overlap = feature_overlap

    @property
    def chunk_instances(self) -> Optional[bool]:
        return self.__chunk_instances

    @chunk_instances.setter
    def chunk_instances(self, chunk_instances: bool):
        assert isinstance(chunk_instances, bool), "The variable chunk_instances needs to be set to true or false"
        self.__chunk_instances = chunk_instances

    @property
    def samples_per_chunk(self) -> Optional[int]:
        return self.__samples_per_chunk

    @samples_per_chunk.setter
    def samples_per_chunk(self, samples_per_chunk) -> None:
        assert self.chunk_instances is True, "In order for samples_per_chunk to be set "
        assert (isinstance(samples_per_chunk, int) and (samples_per_chunk > 0)), "Samples per chunk needs to be a " \
                                                                                 "positive integer."
        self.__samples_per_chunk = samples_per_chunk

        if self.chunk_instances is True:
            assert (isinstance(samples_per_chunk, int) and (
                    int(samples_per_chunk) > 0)), "Samples per chunk needs to be a " \
                                                  "positive integer."
            self.__samples_per_chunk = samples_per_chunk
        else:
            assert (samples_per_chunk is "None"), "If the construct_features attribute is set to False, " \
                                                  "the feature_window attribute needs to be set to \"None\"."
            self.__samples_per_chunk = None

    @property
    def interval_overlap(self) -> Optional[bool]:
        return self.__interval_overlap

    @interval_overlap.setter
    def interval_overlap(self, interval_overlap: bool) -> None:
        assert (isinstance(interval_overlap, bool)), "Interval overlap needs to be a boolean, showing whether when " \
                                                     "chunking the dataset and creating intervals, intermediate " \
                                                     "intervals consisting of 50% of each are to be built."
        self.__interval_overlap = interval_overlap

    @property
    def num_folds(self) -> Optional[int]:
        return self.__num_folds

    @num_folds.setter
    def num_folds(self, num_folds: int) -> None:
        assert int(num_folds) > 0, "Number of folds for cross validation needs to " \
                                                                      "be a positive integer."
        self.__num_folds = num_folds

    def clear_attribute_values(self) -> None:
        """
        Sets each attribute of the sole StudyParameters instance object in use to 'None'.

        """
        print("Setting every attribute in the sole StudyParameter instance to 'None'.")
        for attr, val in vars(self).items():
            self.__setattr__(attr, None)

    # @staticmethod
    # def set_attr(dict):
    #     parameters = StudyParameters()
    #     for attribute, value in dict.items():
    #         if value == "None":
    #             value = None
    #         setattr(parameters, attribute, value)
    #     return parameters


if __name__ == "__main__":
    parameters = StudyParameters()
    parameters.relevant_columns = "[1:102]"
    print("")
