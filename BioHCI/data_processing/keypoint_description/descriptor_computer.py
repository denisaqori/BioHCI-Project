"""
Created: 3/28/19
© Denisa Qori McDonald 2019 All Rights Reserved
"""
import numpy as np
import torch
import os
from BioHCI.data_processing.keypoint_description.interval_descriptor import IntervalDescription
from BioHCI.data_processing.keypoint_description.desc_type import DescType
from BioHCI.data_processing.keypoint_description.sequence_length import SequenceLength
from BioHCI.helpers import utilities as utils
from BioHCI.data.data_constructor import DataConstructor
from BioHCI.helpers.study_config import StudyConfig
from BioHCI.definitions.study_parameters import StudyParameters
from copy import copy
import pickle
from sklearn import preprocessing
import sys
import BioHCI.helpers.type_aliases as types
from typing import Optional, List
from os.path import join
import multiprocessing
import time


class DescriptorComputer:
    def __init__(self, desc_type: DescType, subject_dataset: types.subj_dataset, parameters: StudyParameters,
                 normalize: bool, seq_len: SequenceLength, extra_name: str = "") -> None:

        print("\nProducing dataset descriptors...\n")
        self.desc_type = desc_type
        self.__dataset_descriptors = None

        self.parameters = parameters
        self.normalize = normalize
        self.extra_name = extra_name
        self.seq_len = seq_len

        # self.__dataset_desc_root_path = utils.get_root_path("dataset_desc")
        # if there is no root directory for dataset descriptors, create it
        self.__saved_desc_dir = utils.create_dir(utils.get_root_path("saved_objects") + "/dataset_descriptors")
        # create the full name of the dataset as well, without the path to get there
        self.__dataset_desc_name = self.__produce_dataset_desc_name()
        self.__desc_obj_path = join(self.__saved_desc_dir, self.dataset_desc_name)

        # remove any files remaining from previous tests
        # self.cleanup()

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

        if self.normalize:
            descriptor_subj_dataset = self.normalize_l2(descriptor_subj_dataset)

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

        # check if normalization is to happen
        if self.normalize:
            dataset_desc_name = dataset_desc_name + "_l2_scaled"

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
            dataset_desc = pickle.load(input_file)
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

                if self.desc_type == DescType.JUSD or self.desc_type == DescType.RawData:
                    keypress_split = np.split(keypress, 2, axis=1)
                    normalized_splits = []
                    for split in keypress_split:
                        normalized_split = preprocessing.normalize(split, norm='l2')
                        normalized_splits.append(normalized_split)

                    keypress_normalized = np.concatenate(normalized_splits, axis=1)

                # elif self.desc_type == DescType.MSBSD:
                #     keypress_split = np.split(keypress, [8, 16], axis=1)
                #     normalized_splits = []
                #
                #     for j, split in enumerate(keypress_split):
                #         normalized_split = None
                #         if j == 0 or j == 1:
                #             normalized_split = preprocessing.normalize(split, norm='l2')
                #         elif j == 2:
                #             normalized_split = split
                #         normalized_splits.append(normalized_split)

                    # keypress_normalized = np.concatenate(normalized_splits, axis=1)
                else:
                    print("There is no such descriptor: ", self.desc_type)
                    sys.exit()

                subj_normalized_keypresses.append(keypress_normalized)

            new_subj = copy(subj)
            new_subj.data = subj_normalized_keypresses
            normalized_subj_dataset[subj_name] = new_subj

        return normalized_subj_dataset

    def adjust_sequence_length(self, descriptor_dataset: types.subj_dataset) -> types.subj_dataset:
        """
        Adjusts the sequence length as necessary for each sample in the dataset of descriptors.

        Args:
            descriptor_dataset:

        Returns:

        """
        if self.seq_len == SequenceLength.Existing:
            print(
                f"Only the descriptors discovered are going to be included in the dataset, without any oversampling "
                f"or undersampling. The length of the descriptor sequence that represents each sample may vary.")
        elif self.seq_len == SequenceLength.ZeroPad:
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
        elif self.seq_len == SequenceLength.ExtendEdge:
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
        elif self.seq_len == SequenceLength.Undersample:
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

    def cleanup(self) -> None:
        """
        Removes any files that contain the string "_test" in the dataset descriptors directory.

        Returns: None

        """
        print(f"Deleting any existing files related to dataset descriptor evaluation containing the string '_test'.")
        for filename in os.listdir(self.saved_desc_dir):
            if "_test" in filename:
                full_path_to_remove = join(self.saved_desc_dir, filename)

                os.remove(full_path_to_remove)
                print(f"Deleted file {full_path_to_remove}")


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

    descriptor_computer = DescriptorComputer(DescType.JUSD, subject_dataset, parameters, normalize=True,
                                             seq_len=SequenceLength.ExtendEdge, extra_name="_test")
    descriptors = descriptor_computer.dataset_descriptors
    print("")
