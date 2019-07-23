import os
import numpy as np
import re
import sys
from typing import List, Tuple
from BioHCI.definitions.study_parameters import StudyParameters

labeled_dataset = Tuple[List[np.ndarray], List[str]]


class Subject:

    def __init__(self, subj_data_path: str, parameter: StudyParameters) -> None:

        self.__subj_data_path = subj_data_path
        self.__parameter = parameter

        self.__filename_list = []

        self.__data, self.__categories = self.__build_subj_data()

        assert len(self.__data) == len(self.__categories), \
            "The sizes of the subject's data list and categories list do not match!!"
        self.__all_data_bool = True

    def __build_subj_data(self) -> labeled_dataset:
        """
        Extracts signal data with its corresponding categories for the subject, found it the subject's files.

        Returns:
            subj_category_data (List[np.ndarray]): (a python list of numpy arrays with all the signal data from the text
                                                   files of one subject.)
            subj_category_names (List[str]): a python list of stings, corresponding to the category names of the data in
                                             each element of subj_category_data.
        """
        print("\nBuilding the subject dataset: ")
        subj_category_data = []
        subj_category_names = []

        # if each subject has a directory for each category
        if self.__parameter.cat_names == 'dir':
            for cat_data_container in os.listdir(self.__subj_data_path):
                subj_cat_data_path = os.path.join(self.__subj_data_path, cat_data_container)
                if os.path.isdir(subj_cat_data_path):
                    category_data, category_names = self.__read_files(subj_cat_data_path, cat_data_container)
                    subj_category_data.append(category_data)
                    subj_category_names.append(category_names)

            subj_category_data = [item for sublist in subj_category_data for item in sublist]
            subj_category_names = [item for sublist in subj_category_names for item in sublist]

        # if each subject has one file per category
        elif self.__parameter.cat_names == 'file':
            subj_cat_data_path = self.__subj_data_path
            subj_category_data, subj_category_names = self.__read_files(subj_cat_data_path)
        else:
            print(
                "A problem with assigning category names. They should be assigned based on filenames or "
                "directory names. Exiting...")
            sys.exit()

        return subj_category_data, subj_category_names

    def __read_files(self, dirpath: str, label=None) -> labeled_dataset:
        """
        Given a directory, find files

        Args:
            dirpath:
            label:

        Returns:

        """
        data = []
        labels = []

        for filename in os.listdir(dirpath):
            if filename.endswith(self.__parameter.file_format):

                # split the filename into the name part and the extension part
                name, extension = os.path.splitext(filename)

                filepath = os.path.join(dirpath, filename)
                filedata = self.__get_file_data(filepath)

                data.append(filedata)
                if label is None:
                    labels.append(name)
                else:
                    labels.append(label)

        return data, labels

    def __get_file_data(self, filepath: str) -> np.ndarray:
        """
        Obtain the data in the give file, potentially filtering out some rows and columns as determined by
        self.__parameter.

        Args:
            filepath (str): the path to the file whose data is to be extracted

        Returns:
            file_lines (np.ndarray): the extracted data from the given file

        """
        with open(filepath, encoding='ascii') as f:
            # get the data in each file by first stripping and splitting the lines and
            # then creating a numpy array out of these values
            file_lines = []
            print("Filename: ", filepath)
            for line in f:
                line = line.strip(' \t\n\r')
                line = re.split('\t|,', line)
                file_lines.append(line)
            file_lines = np.asarray(file_lines)

            # keep info only from the relevant columns and rows
            file_lines = (
                file_lines[self.__parameter.start_row:, self.__parameter.relevant_columns]).astype(np.float32)
            return file_lines

    @property
    def data(self) -> List[np.ndarray]:
        """
        Returns: subject data split by categories, as a list of numpy arrays
        """
        return self.__data

    @data.setter
    def data(self, data: List[np.ndarray]) -> None:
        """
        Set the subject's data to the passed argument
        Args:
            data (list[np.ndarray]):
        """
        self.__data = data

    @property
    def categories(self) -> List[str]:
        """
        Returns: subject categories
        """
        return self.__categories

    @categories.setter
    def categories(self, categories: List[str]) -> None:
        """
        Sets the subject categories to the passed 'categories' argument.

        Args:
            categories: List of categories, where each element belongs to a an element in the data property

        """
        self.__categories = categories

    @property
    def all_data_bool(self) -> bool:
        return self.__all_data_bool

    @all_data_bool.setter
    def all_data_bool(self, all_data_bool: bool) -> None:
        self.__all_data_bool = all_data_bool
