"""
Created: 3/28/19
© Denisa Qori McDonald 2019 All Rights Reserved
"""
import multiprocessing
import os
import pickle
import time
from copy import copy
from os.path import join
from typing import List, Optional

import numpy as np
import torch

import BioHCI.helpers.type_aliases as types
from BioHCI.data.data_constructor import DataConstructor
from BioHCI.data_processing.keypoint_description.desc_type import DescType
from BioHCI.data_processing.keypoint_description.interval_descriptor import IntervalDescription
from BioHCI.data_processing.keypoint_description.sequence_length import SeqLen
from BioHCI.definitions.study_parameters import StudyParameters
from BioHCI.helpers import utilities as utils
from BioHCI.helpers.study_config import StudyConfig


class DescriptorComputer:
    def __init__(self, desc_type: DescType, subject_dataset: types.subj_dataset, parameters: StudyParameters,
                 seq_len: SeqLen, extra_name: str = "") -> None:

        print("\nProducing dataset descriptors...\n")
        self.desc_type = desc_type
        self.__dataset_descriptors = None

        self.parameters = parameters
        self.extra_name = extra_name
        self.seq_len = seq_len

        # self.__dataset_desc_root_path = utils.get_root_path("dataset_desc")
        # if there is no root directory for dataset descriptors, create it
        saved_obj_subdir = self.parameters.study_name + "/dataset_descriptors"
        self.__saved_desc_dir = utils.create_dir(join(utils.get_root_path("saved_objects"), saved_obj_subdir))

        # create the full name of the dataset as well, without the path to get there
        self.__dataset_desc_name = self.__produce_dataset_desc_name()
        self.__desc_obj_path = join(self.__saved_desc_dir, self.dataset_desc_name)

        # remove any files remaining from previous tests
        # utils.cleanup(self.saved_desc_dir, "_test")

        # create the full path to save the current descriptor if it does not exist, or to load from if it does
        if os.path.exists(self.desc_obj_path):
            print("Loading dataset descriptors from: ", self.desc_obj_path)
            self.__dataset_descriptors = self.load_descriptors(self.desc_obj_path)
        else:
            print("Computing dataset descriptors...")
            self.__dataset_descriptors = self.produce_dataset_descriptors(subject_dataset)
        print("")

    @property
    def dataset_desc_name(self) -> Optional[str]:
        return self.__dataset_desc_name

    @property
    def saved_desc_dir(self) -> Optional[str]:
        return self.__saved_desc_dir

    @property
    def desc_obj_path(self):
        return self.__desc_obj_path

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
        if self.desc_type == DescType.RawData:
            descriptor_subj_dataset = subject_dataset

        else:
            num_processes = multiprocessing.cpu_count() * 2
            start_time = time.time()

            descriptor_subj_dataset = {}
            for subj_name, subj in subject_dataset.items():
                subj_data = subj.data

                print(f"Total number of descriptor sets to compute: {len(subj_data)}")
                with multiprocessing.Pool(processes=num_processes) as pool:
                    subj_keypress_desc = pool.map(self.produce_subj_keypress_descriptors, subj_data)
                pool.close()
                pool.join()

                # put lists in proper format
                subj_keypress_desc = [desc for sublist in subj_keypress_desc for desc in sublist]

                new_subj = copy(subj)
                new_subj.data = subj_keypress_desc
                descriptor_subj_dataset[subj_name] = new_subj

            duration_with_pool = utils.time_since(start_time)
            print(f"\nComputed dataset descriptors for {len(subject_dataset)} subject(s), using {num_processes} "
                  f"processes, for a duration of {duration_with_pool}")

        descriptor_subj_dataset = self.adjust_sequence_length(descriptor_subj_dataset)

        self.save_descriptors(descriptor_subj_dataset)
        return descriptor_subj_dataset

    def produce_subj_keypress_descriptors(self, keypress: np.ndarray) -> List[np.ndarray]:
        """

        Args:
            keypress:

        Returns:

        """
        # print(multiprocessing.current_process())
        interval_desc_list = IntervalDescription(keypress, self.desc_type).descriptors

        return interval_desc_list

    def __produce_dataset_desc_name(self):
        """
        Creates the name of the dataset descriptor based on its characteristics, as well as the path it is to be stored.

        Returns:
            dataset_desc_path, dataset_desc_name (str, str): the path and the name of the dataset descriptor object.

        """
        dataset_desc_name = self.parameters.study_name + "_" + str(self.desc_type) + "_" + str(self.seq_len)

        # check if there is an extra name to add to the existing descriptor name
        if self.extra_name is not None:
            dataset_desc_name = dataset_desc_name + self.extra_name

        # dataset_desc_path = join(self.__dataset_desc_root_path, dataset_desc_name + ".pkl")
        return dataset_desc_name

    @staticmethod
    def load_descriptors(desc_obj_path: str) -> types.subj_dataset:
        """
        Loads descriptors from a pickled object.

        Args:
            desc_obj_path (path): path of the file where object to be loaded is stored.


        Returns:
            dataset_desc (dict): a dictionary mapping a subject name to as Subject object,
                whose data is comprised of its descriptors for each category.

        """
        with open(desc_obj_path, "rb") as input_file:
            dataset_desc = pickle.load(input_file, encoding="bytes")
        return dataset_desc

    def save_descriptors(self, descriptors: types.subj_dataset) -> None:
        """
        Saves the computed dataset descriptors

        Args:
            descriptors (dict): a dictionary mapping a subject name to his/her dataset of descriptors
        """
        print(f"\nSaving dataset descriptors to {self.desc_obj_path}")
        if not os.path.exists(self.desc_obj_path):
            with open(self.desc_obj_path, 'wb') as f:
                pickle.dump(descriptors, f, pickle.HIGHEST_PROTOCOL)

    def adjust_sequence_length(self, descriptor_dataset: types.subj_dataset) -> types.subj_dataset:
        """
        Adjusts the sequence length as necessary for each sample in the dataset of descriptors.

        Args:
            descriptor_dataset:

        Returns:

        """
        if self.seq_len == SeqLen.Existing:
            print(
                f"Only the descriptors discovered are going to be included in the dataset, without any oversampling "
                f"or undersampling. The length of the descriptor sequence that represents each sample may vary.")
        elif self.seq_len == SeqLen.ZeroPad:
            print(
                f"Padding samples with zeros so that each sequence has the same desriptor length. ")

            max_len = 0
            for subj_name, subject in descriptor_dataset.items():
                for i, sample in enumerate(subject.data):
                    if sample.shape[0] > max_len:
                        max_len = sample.shape[0]

            for subj_name, subject in descriptor_dataset.items():
                for i, sample in enumerate(subject.data):
                    num_rows_to_add = max_len - sample.shape[0]
                    subject.data[i] = np.pad(subject.data[i], [(0, num_rows_to_add), (0, 0)], mode='constant',
                                             constant_values=0)
        elif self.seq_len == SeqLen.ExtendEdge:
            print(
                f"Padding samples with the last descriptor so that each sequence has the same desriptor length. ")

            max_len = 0
            for subj_name, subject in descriptor_dataset.items():
                for i, sample in enumerate(subject.data):
                    if sample.shape[0] > max_len:
                        max_len = sample.shape[0]

            for subj_name, subject in descriptor_dataset.items():
                for i, sample in enumerate(subject.data):
                    num_rows_to_add = max_len - sample.shape[0]
                    subject.data[i] = np.pad(subject.data[i], [(0, num_rows_to_add), (0, 0)], mode='edge')
        elif self.seq_len == SeqLen.Undersample:
            print(f"Padding samples so that each sequence has the same descriptor length. ")

            min_len = 200
            for subj_name, subject in descriptor_dataset.items():
                for i, sample in enumerate(subject.data):
                    if sample.shape[0] < min_len:
                        min_len = sample.shape[0]

            for subj_name, subject in descriptor_dataset.items():
                for i, sample in enumerate(subject.data):
                    subject.data[i] = subject.data[i][0:min_len]
        else:
            print("Sequence length type undefined. Returning original dataset.")

        return descriptor_dataset


if __name__ == "__main__":
    print("Running feature_constructor module...")
    print("Is cuda available?", torch.cuda.is_available())

    config_dir = "config_files"
    config = StudyConfig(config_dir)

    # create a template of a configuration file with all the fields initialized to None
    config.create_config_file_template()
    # parameters = config.populate_study_parameters("CTS_Keyboard_simple.toml")
    parameters = config.populate_study_parameters("CTS_5taps_per_button.toml")

    # generating the data from files
    data = DataConstructor(parameters)
    subject_dataset = data.get_subject_dataset()

    descriptor_computer = DescriptorComputer(DescType.MSD, subject_dataset, parameters, seq_len=SeqLen.ExtendEdge,
                                             extra_name="_test2")
    descriptors = descriptor_computer.dataset_descriptors
    print("")
