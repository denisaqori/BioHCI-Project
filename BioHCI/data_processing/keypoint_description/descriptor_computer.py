"""
Created: 3/28/19
© Denisa Qori McDonald 2019 All Rights Reserved
"""
import numpy as np
import torch
import os
from BioHCI.data_processing.keypoint_description.interval_descriptor import IntervalDescription
from BioHCI.data_processing.keypoint_description.desc_type import DescType
from BioHCI.helpers import utilities as utils
from BioHCI.data.data_constructor import DataConstructor
from BioHCI.helpers.study_config import StudyConfig
from BioHCI.definition.study_parameters import StudyParameters
from copy import copy
import pickle
from sklearn import preprocessing
import sys
import BioHCI.helpers.type_aliases as types
from typing import Optional
from os.path import join


class DescriptorComputer:
    def __init__(self, desc_type: DescType, subject_dataset: types.subj_dataset, parameters: StudyParameters,
                 normalize: bool, extra_name: str = "") -> None:

        self.desc_type = desc_type
        self.__dataset_descriptors = None

        assert isinstance(parameters, StudyParameters)
        self.parameters = parameters

        assert isinstance(normalize, bool)
        self.normalize = normalize

        assert isinstance(extra_name, str)
        self.extra_name = extra_name

        self.__dataset_desc_root_path = utils.get_root_path("dataset_desc")
        # if there is no root directory for dataset descriptors, create it
        utils.create_dir(self.__dataset_desc_root_path)
        # create the full name of the dataset as well, without the path to get there
        self.__dataset_desc_path, self.__dataset_desc_name = self.__produce_dataset_desc_path_and_name()

        # create the full path to save the current descriptor if it does not exist, or to load from if it does
        if os.path.exists(self.dataset_desc_path):
            print("Loading dataset descriptors from: ", self.dataset_desc_path)
            self.__dataset_descriptors = self.load_descriptors(self.dataset_desc_path)
        else:
            print("Producing dataset descriptors...")
            self.__dataset_descriptors = self.produce_dataset_descriptors(subject_dataset)
        print("")

    @property
    def dataset_desc_name(self) -> Optional[str]:
        return self.__dataset_desc_name

    @property
    def dataset_desc_path(self) -> Optional[str]:
        return self.__dataset_desc_path

    @property
    def dataset_descriptors(self) -> Optional[types.subj_dataset]:
        return self.__dataset_descriptors

    def produce_dataset_descriptors(self, subject_dataset: types.subj_dataset) -> types.subj_dataset:
        """
        Produces and saves to a file the descriptors for the subject dataset.

        Args:
            subject_dataset (dict): dictionary mapping a subject name to a Subject object.

        Returns:
            descriptor_subj_dataset (dict): a dictionary mapping a subject name to as Subject object,
                whose data is comprised of its descriptors for each category.
        """
        descriptor_subj_dataset = {}
        for subj_name, subj in subject_dataset.items():
            subj_data = subj.data
            subj_keypress_desc = []
            for i, keypress in enumerate(subj_data):
                interval_desc_list = IntervalDescription(keypress, self.desc_type).descriptors
                subj_keypress_desc.append(interval_desc_list)

            subj_keypress_desc = [desc for sublist in subj_keypress_desc for desc in sublist]
            new_subj = copy(subj)
            new_subj.data = subj_keypress_desc
            descriptor_subj_dataset[subj_name] = new_subj

        if self.normalize:
            descriptor_subj_dataset = self.normalize_l2(descriptor_subj_dataset)

        self.save_descriptors(descriptor_subj_dataset)
        return descriptor_subj_dataset

    def __produce_dataset_desc_path_and_name(self):
        """
        Creates the name of the dataset descriptor based on its characteristics, as well as the path it is to be stored.

        Returns:
            dataset_desc_path, dataset_desc_name (str, str): the path and the name of the dataset descriptor object.

        """
        dataset_desc_name = self.parameters.study_name + "_" + str(self.desc_type)

        # check if normalization is to happen
        if self.normalize:
            dataset_desc_name = dataset_desc_name + "_l2_scaled"

        # check if there is an extra name to add to the existing descriptor name
        if self.extra_name is not None:
            dataset_desc_name = dataset_desc_name + self.extra_name

        dataset_desc_path = join(self.__dataset_desc_root_path, dataset_desc_name + ".pkl")
        dataset_desc_name = dataset_desc_name
        return dataset_desc_path, dataset_desc_name

    @staticmethod
    def load_descriptors(dataset_desc_path: str) -> types.subj_dataset:
        """
        Loads descriptors from a pickled object.

        Args:
            dataset_desc_path (path): path of the file where object to be loaded is stored.

        Returns:
            dataset_desc (dict): a dictionary mapping a subject name to as Subject object,
                whose data is comprised of its descriptors for each category.

        """
        with open(dataset_desc_path, "rb") as input_file:
            dataset_desc = pickle.load(input_file)
        return dataset_desc

    def save_descriptors(self, descriptors: types.subj_dataset) -> None:
        """
        Saves the computed dataset descriptors

        Args:
            descriptors (dict): a dictionary mapping a subject name to his/her dataset of descriptors
        """
        if not os.path.exists(self.__dataset_desc_path):
            with open(self.__dataset_desc_path, 'wb') as f:
                pickle.dump(descriptors, f, pickle.HIGHEST_PROTOCOL)

    def normalize_l2(self, dataset_desc: types.subj_dataset):
        """
        If the type of descriptor is JUSD, normalizes each instance of dataset_desc - converts each row into unit norm.
        If the type of descriptor is MSBSD, first splits each row in half, normalized each half row, and then puts them
        back together.

        Args:
            dataset_desc:

        Returns:

        """
        normalized_subj_dataset = {}
        for subj_name, subj in dataset_desc.items():
            subj_data = subj.data
            subj_normalized_keypresses = []
            for i, keypress in enumerate(subj_data):

                if self.desc_type == DescType.JUSD:
                    keypress_normalized = preprocessing.normalize(keypress, norm='l2')

                elif self.desc_type == DescType.MSBSD:
                    keypress_split = np.split(keypress, 2, axis=1)
                    normalized_splits = []
                    for split in keypress_split:
                        normalized_split = preprocessing.normalize(split, norm='l2')
                        normalized_splits.append(normalized_split)
                    keypress_normalized = np.concatenate(normalized_splits, axis=1)
                else:
                    print("There is no such descriptor: ", self.desc_type)
                    sys.exit()

                subj_normalized_keypresses.append(keypress_normalized)

            new_subj = copy(subj)
            new_subj.data = subj_normalized_keypresses
            normalized_subj_dataset[subj_name] = new_subj

        return normalized_subj_dataset


if __name__ == "__main__":
    print("Running feature_constructor module...")
    print("Is cuda available?", torch.cuda.is_available())

    config_dir = "config_files"
    config = StudyConfig(config_dir)

    # create a template of a configuration file with all the fields initialized to None
    config.create_config_file_template()
    parameters = config.populate_study_parameters("CTS_5taps_per_button.toml")

    # generating the data from files
    data = DataConstructor(parameters)
    subject_dataset = data.get_subject_dataset()

    descriptor_computer = DescriptorComputer(DescType.JUSD, subject_dataset, parameters, normalize=True,
                                             extra_name="_test")
    all_desc = descriptor_computer.produce_dataset_descriptors(subject_dataset)
