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


class DescriptorComputer:
    def __init__(self, desc_type: DescType, parameters: StudyParameters, normalize: bool, dataset_desc_name: str = ""):

        self.desc_type = desc_type

        # set name and path of descriptor to be saved
        self.__dataset_desc_path = None
        assert isinstance(dataset_desc_name, str)
        self.dataset_desc_name = dataset_desc_name
        dataset_desc_path = './BioHCI/data_processing/keypoint_description/dataset_descriptors'
        self.all_dataset_desc_dir = utils.create_dir(dataset_desc_path)

        assert isinstance(normalize, bool)
        self.normalize = normalize

        assert isinstance(parameters, StudyParameters)
        self.parameters = parameters

    @property
    def dataset_desc_path(self) -> str:
        return self.__dataset_desc_path

    def produce_dataset_descriptors(self, subject_dataset: types.subj_dataset):
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

        self.save_to_file(descriptor_subj_dataset)
        return descriptor_subj_dataset

    def save_to_file(self, obj) -> str:
        """
        Returns the path to the numpy array containing the dataset description whose name is passed as an
        argument

        Args:
        dataset_desc_name: The name of the dataset descriptors whose path is to be returned

        Returns:
            dataset_desc_path: the absolute path to that numpy array containing dataset descriptors
        """
        dataset_desc_path = os.path.abspath(os.path.join(self.all_dataset_desc_dir, self.parameters.study_name +
                                                         "_desc_type_" + str(self.desc_type)))
        if self.dataset_desc_name is not None:
            dataset_desc_path = dataset_desc_path + self.dataset_desc_name + ".pkl"
        else:
            dataset_desc_path = dataset_desc_path + ".pkl"

        if not os.path.exists(dataset_desc_path):
            with open(dataset_desc_path, 'wb') as f:
                pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

        self.__dataset_desc_path = dataset_desc_path
        return dataset_desc_path

    def normalize_l2(self, dataset_desc):
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

        self.dataset_desc_name += "_l2_scaled"
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

    descriptor_computer = DescriptorComputer(DescType.JUSD, parameters, normalize=True, dataset_desc_name="_test")
    all_desc = descriptor_computer.produce_dataset_descriptors(subject_dataset)
