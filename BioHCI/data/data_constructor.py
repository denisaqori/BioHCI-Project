import os
import pprint as pp
import re
import sys
import numpy as np

from scipy import stats

from BioHCI.data.subject import Subject
from BioHCI.definitions.study_parameters import StudyParameters
from BioHCI.helpers.study_config import StudyConfig
from typing import List, Dict
import BioHCI.helpers.type_aliases as types


# TODO: add standardization option within feature for every subject - maybe entails building a dataframe first and
# then splitting again
class DataConstructor:
    def __init__(self, parameters: StudyParameters) -> None:

        self.__parameters = parameters

        # number of files to be included in the dataset where each file contains fNIRS data for one subject
        self.__num_subj = parameters.num_subj

        # parameter's resource_path property determines the location of the data files, to be found within separate
        # directories for each subject. These are sub-directories of this path.
        self.__dir_path = parameters.resource_path

        self.subj_dir_list = self.get_subj_dir_list()
        assert self.__num_subj == len(self.subj_dir_list), "Not as many subject directories found as declared in " \
                                                           "StudyParameters"
        print("List of sub-directory names from where data will be obtained for each subject: ", self.subj_dir_list)

        self._subject_identity_list = self.create_subject_identity_list(self.subj_dir_list)
        print("\nSubject List: ", self._subject_identity_list, "\n")

        self.subj_dataset = self.build_all_subj_dataset(self.subj_dir_list)

        self.print_all_subj_dataset()

    # this method determines the path to the directory with each subject's data files and returns a list of names of
    # directories which should be of each subject
    def get_subj_dir_list(self) -> List[str]:
        # we start by iterating through each .txt/.csv file in the given path
        directory = os.path.abspath(os.path.join(os.pardir, self.__dir_path))

        # we start a list where we can store the file names we are using,
        # since a lot of labeling depends on each file
        subj_dir_list = []

        for subj_dir in os.listdir(directory):
            dir_name = os.fsdecode(subj_dir)
            subj_dir_list.append(dir_name)

        return subj_dir_list

    # this method returns a Python dictionary of Subjects, with the key being the subject number and value being the
    # Subject object. Each Subject contains his/her own data split by categories
    # (one or more). The data itself can be accessed by the calling the Subject class methods.
    def build_all_subj_dataset(self, subj_dir_list: List[str]) -> types.subj_dataset:

        all_subj: Dict[str, Subject] = {}

        # for each subject directory, create the whole path and give that to the Subject class
        # in order for it to build the dataset from files found there
        for subj_dir_name in subj_dir_list:
            subj_data_path = os.path.join(self.__dir_path, subj_dir_name)

            subj = Subject(subj_data_path, self.__parameters)
            all_subj[subj_dir_name] = subj

        return all_subj

    def print_all_subj_dataset(self) -> None:
        """
        Print and plot information about the subject dataset -  subject names and data shapes per category

        Returns: None

        """
        print("\nSubject dataset:")
        for subj_name, subj in self.subj_dataset.items():
            print("Subj", subj_name, "- shapes of each category data with the corresponding categories:")

            for i, cat_data in enumerate(subj.data):
                pp.pprint(cat_data.shape)

            pp.pprint(subj.categories)
            print("\n")

    # create a subject identity list, where each subject is an integer, and the list is sorted
    @staticmethod
    def create_subject_identity_list(subj_dir_list: List[str]) -> List[int]:
        subject_list = []
        for subj_dir_name in subj_dir_list:
            # remove non-number characters from directory name
            subj_dir_name = re.sub("\D", "", subj_dir_name)
            try:
                subj = int(subj_dir_name)
            except ValueError:
                print("Sub-directories must be named by the number of the subject only")
                print("Exiting...")
                sys.exit()
            subject_list.append(subj)

        subject_list.sort()
        return subject_list

    # this method standardizes the dataset that is passed to it. The format of the dataset variable is assumed to be
    # the same as that of index_labeled_dataset: a python list of numpy arrays of shape
    # ((number of subjects in category) x (number of instances per file/subject) x (number of features))
    # TODO: needs to change to fit the new structure of the dataframe
    @staticmethod
    def standardize(dataset, std_type):
        standardized_dataset = []
        if std_type is 'PerSubjPerColumn':
            print("\nCreating a standardized (z-score) dataset by subject and channel...\n")
            for j, category_data in enumerate(dataset):
                # standardize across the second axis (inst_per_subj)
                standardized_category = stats.zscore(category_data, axis=1)
                standardized_dataset.append(standardized_category)
        elif std_type is 'perSubj':
            print("Creating a standardized (z-score) dataset by subject only...\n")
            for j, category_data in enumerate(dataset):
                # standardize the whole array since no axis is specified
                standardized_category = stats.zscore(category_data)
                standardized_dataset.append(standardized_category)
        else:
            print("Standardization method defined in AbstractData not implemented!")
            print("Exiting...")
            sys.exit()

        return standardized_dataset

    def get_subject_dataset(self) -> types.subj_dataset:
        return self.subj_dataset

    # this function returns the set of all categories found across all subjects in the study (all unique values)
    def get_all_dataset_categories(self) -> List[str]:
        category_list = []
        for subj_name, subj in self.subj_dataset.items():
            category_list.append(subj.categories)

        flat_list = [item for sublist in category_list for item in sublist]
        categories = list(set(flat_list))

        # return the unique values in flat_list
        return categories

    @staticmethod
    def fft_domain(cat):
        freq_spect = np.fft.fft(cat, axis=0)
        return freq_spect


if __name__ == "__main__":
    config_dir = "config_files"
    config = StudyConfig(config_dir)

    # create a template of a configuration file with all the fields initialized to None
    config.create_config_file_template()

    # the object with variable definitions based on the specified configuration file. It includes data description,
    # definitions of run parameters (independent of deep definitions vs not)
    parameters = config.populate_study_parameters("CTS_one_subj_variable.toml")

    data = DataConstructor(parameters)
    subject_dict = data.get_subject_dataset()
