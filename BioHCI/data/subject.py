import os
import numpy as np
import re
import sys
from typing import List, Tuple

from numpy.polynomial import Polynomial

from BioHCI.definitions.study_parameters import StudyParameters
import scipy.stats as stats
labeled_dataset = Tuple[List[np.ndarray], List[str]]
import matplotlib.pyplot as plt

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
        subj_category_data = []
        subj_category_names = []

        subj_baseline_container = None
        # if each subject has a directory for each category
        if self.__parameter.cat_names == 'dir':

            # if there is baseline data for the subject, get its path
            for cat_data_container in os.listdir(self.__subj_data_path):
                if cat_data_container == "button000":
                    subj_baseline_container = os.path.join(self.__subj_data_path, cat_data_container)

            # go to each category directory of the subject
            for cat_data_container in os.listdir(self.__subj_data_path):
                if cat_data_container != "button000":
                    subj_cat_data_path = os.path.join(self.__subj_data_path, cat_data_container)
                    if os.path.isdir(subj_cat_data_path):
                        category_data, category_names = self.__read_files(subj_cat_data_path,
                                                                          subj_baseline_container, cat_data_container)
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

    def __read_files(self, data_dirpath: str, baseline_dirpath = None, label = None) -> labeled_dataset:
        """
        Given a directory, find files

        Args:
            dirpath:
            label:

        Returns:

        """
        all_baseline_fnames = []
        if baseline_dirpath is not None:
            for baseline_fname in os.listdir(baseline_dirpath):
                if baseline_fname.endswith(self.__parameter.file_format) and "trial" in baseline_fname:
                    all_baseline_fnames.append(baseline_fname)

        data = []
        labels = []

        for filename in os.listdir(data_dirpath):
            if filename.endswith(self.__parameter.file_format):

                # split the filename into the name part and the extension part
                name, extension = os.path.splitext(filename)
                filepath = os.path.join(data_dirpath, filename)
                filedata = self.__get_file_data(filepath)

                if baseline_dirpath is not None:
                    # find the baseline file corresponding to the trial number of the data
                    trial_num_str = name[name.find("_trial"): name.find("_button")+1]
                    baseline_fname = None
                    for fname in all_baseline_fnames:
                        if trial_num_str in fname:
                            baseline_fname = fname
                            break
                    # get the baseline data
                    baseline_filepath = os.path.join(baseline_dirpath, baseline_fname)
                    baseline_data = self.__get_file_data(baseline_filepath)

                    assert baseline_data.shape == filedata.shape
                    filedata = filedata - baseline_data

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
            # print("Filename: ", filepath)
            for line in f:
                line = line.strip(' \t\n\r')
                line = re.split('\t|,', line)
                file_lines.append(line)
            file_lines = np.asarray(file_lines)

            # keep info only from the relevant columns and rows
            file_lines = (
                file_lines[self.__parameter.start_row:, self.__parameter.relevant_columns]).astype(np.float32)

            # calculating average of all frequencies for each signal
            stat_array = self.get_attribute_stats(file_lines)
            return stat_array
            # return file_lines

    @staticmethod
    def get_attribute_stats(single_file_dataset):

        # return values of odd columns
        odd_column_array = single_file_dataset[:, 1::2]
        even_column_array = single_file_dataset[:, 0::2]

        odd_mean = np.mean(odd_column_array, axis=1)
        odd_mean = np.expand_dims(odd_mean, axis=1)
        # return values of even columns
        even_mean = np.mean(even_column_array, axis=1)
        even_mean = np.expand_dims(even_mean, axis=1)

        odd_std = np.expand_dims(np.std(odd_column_array, axis=1), axis=1)
        even_std = np.expand_dims(np.std(even_column_array, axis=1), axis=1)

        odd_sum = np.expand_dims(np.sum(odd_column_array, axis=1), axis=1)
        even_sum = np.expand_dims(np.sum(even_column_array, axis=1), axis=1)

        # """
        # fitting a function to all frequnecies for each time step
        x = np.arange(0, odd_column_array.shape[1])
        odd_linreg_ls = []
        odd_coef_ls = []
        even_linreg_ls = []
        even_coef_ls = []
        for i in range(0, odd_column_array.shape[0]):
            odd_time_step = odd_column_array[i, :]
            even_time_step = even_column_array[i, :]

            # plt.plot(x, odd_time_step)
            # plt.plot(x, even_time_step)
            # plt.show()

            odd_slope, odd_intercept, odd_r_value, odd_p_value, odd_std_err = stats.linregress(x, odd_time_step)
            odd_linreg = [odd_slope, odd_intercept, odd_r_value, odd_p_value, odd_std_err]

            odd_coef = Polynomial.fit(x, odd_time_step, deg=2)
            odd_coef = odd_coef.coef.tolist()
            odd_coef_ls.append(odd_coef)

            odd_linreg_ls.append(odd_linreg)

            even_slope, even_intercept, even_r_value, even_p_value, even_std_err = stats.linregress(x, even_time_step)
            even_linreg = [even_slope, even_intercept, even_r_value, even_p_value, even_std_err]

            even_coef = Polynomial.fit(x, even_time_step, deg=2)
            even_coef = even_coef.coef.tolist()
            even_coef_ls.append(even_coef)

            even_linreg_ls.append(even_linreg)

        odd_linreg_stats = np.array([np.array(xi) for xi in odd_linreg_ls])
        even_linreg_stats = np.array([np.array(xi) for xi in even_linreg_ls])
        odd_coef_stats = np.array([np.array(xi) for xi in odd_coef_ls])
        even_coef_stats = np.array([np.array(xi) for xi in even_coef_ls])
        # """
        stat_array = np.concatenate((odd_mean, even_mean, odd_std, even_std, odd_sum, even_sum, odd_linreg_stats,
                                     even_linreg_stats), axis=1)
        return stat_array

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
