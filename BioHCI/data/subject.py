import os
import numpy as np
import re
import sys
from typing import List, Tuple, Optional

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
        print(f"Number of data points is {len(self.__data)}, and number of corresponding categories is {len(self.__categories)}.")

        assert len(self.__data) == len(self.__categories), \
            "The sizes of the subject's data list and categories list do not match!!"
        self.__all_data_bool = True

    # TODO: baseline folder name - unify
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

    def __read_files(self, data_dirpath: str, baseline_dirpath=None, label=None) -> labeled_dataset:
        """
        Given a directory, find files

        Args:
            data_dirpath: The data main directory
            baseline_dirpath: If provided, it is the directory containing all baseline data
            label: If provided, the label should be given to data from all files in that directory

        Returns:
            data: all file data
            labels: all corresponding labels to the data
        """
        data = []
        labels = []

        for filename in os.listdir(data_dirpath):
            if filename.endswith(self.__parameter.file_format) and "baseline" not in filename:

                # split the filename into the name part and the extension part
                fname, extension = os.path.splitext(filename)
                filepath = os.path.join(data_dirpath, filename)
                filedata = self.__get_file_data(filepath)

                if baseline_dirpath is not None:
                    baseline_dict = self.__get_baseline_data(structure="dir", baseline_dirpath=baseline_dirpath,
                                                             name=fname)
                else:
                    baseline_dict = self.__get_baseline_data(structure="file", data_dirpath=data_dirpath)

                if baseline_dict is not None:
                    file_collection_num = self.__get_number_index(fname, "_collection", 3)

                    # find the right baseline file for the current data file (depending on "collection")
                    for baseline_name, baseline_data in baseline_dict.items():
                        baseline_collection_name = self.__get_number_index(baseline_name, "_collection", 3)

                        if file_collection_num == baseline_collection_name:

                            # subtract the baseline data from the event data
                            if baseline_data.shape == filedata.shape:
                                filedata = filedata - baseline_data
                                data.append(filedata)
                                if label is None:
                                    labels.append(fname)
                                else:
                                    labels.append(label)
                                break
                            else:
                                print(
                                    f"There is a problem with the number of rows or columns of either {filename} or its "
                                    f"baseline data. Removing file from processing.")

        return data, labels

    @staticmethod
    def __get_number_index(main_string: str, sub_string: str, num_len: int = 3) -> str:
        """
        Finds a number in the string that is situated immediately after a particular substring. Useful for selecting
        conditions by number in a filename.

        Args:
            main_string: The string to search (typically the filename)
            sub_string: The substring after which the number is located (typically condition we are looking for)
            num_len: The length of the number, the number of digits.

        Returns: The string version of the number that should be situated right after the substring in a string.

        """
        idx = main_string.find(sub_string)
        assert idx != -1, f"String {main_string} does not contain substring {sub_string}"
        start_idx = idx + len(sub_string)
        num_str = main_string[start_idx: start_idx + num_len]

        assert int(num_str), f"The string {num_str} is not a number - something is off with the number of " \
                             f"digits being searched or the substring after which this number is found."
        return num_str

    def __get_baseline_data(self, structure: Optional[str] = None, baseline_dirpath: Optional[str] = None,
                            name: Optional[str] = None, data_dirpath: Optional[str] = None) -> Optional[dict]:
        """
        Returns the data from a baseline file.

        Args:
            structure: Indicates how the baseline data is structured: whether all is placed in one directory ("dir"),
                       or whether a baseline file is placed within each data directory ("file").
            baseline_dirpath: If structure is "dir", this indicates the baseline directory. Otherwise, it should be left
                              as None.
            name: If structure is "dir", it indicates the filename whose baseline to search in the baseline directory.
                  Otherwise it should be left as None.
            data_dirpath: If structure is "file" it indicates the data directory where we are searching for a baseline
                          file to subtract from each other element of the dataset. Otherwise, it should be left as None.

        Returns:
            baseline_data: the baseline data
        """
        # baseline_dict = None
        assert structure == "dir" or structure == "file" or structure is None, \
            "Potential typo, or unimplemented baseline organization."

        baseline_dict = {}
        if structure == "file":
            assert baseline_dirpath is None and name is None, "They are only need for 'dir' structure."
            assert data_dirpath is not None
            for filename in os.listdir(data_dirpath):
                if filename.endswith(self.__parameter.file_format):
                    if "baseline" in filename:
                        b_filepath = os.path.join(data_dirpath, filename)
                        baseline_data = self.__get_file_data(b_filepath)
                        baseline_dict[filename] = baseline_data
                        # break
        else:
            assert structure is None
            print(f"No baseline found. Returning None. Might be \"dir\" structure commented out currently for "
                  f"compatibility issues.")

            # need to double-check it works with folder structure when necessary
            """"
        elif structure == "dir":
            all_baseline_fnames = []
            assert baseline_dirpath is not None and name is not None, \
                "If all baselines are found in one directory, that directory needs to be passed as an argument."
            for baseline_fname in os.listdir(baseline_dirpath):
                if baseline_fname.endswith(self.__parameter.file_format) and "trial" in baseline_fname:
                    all_baseline_fnames.append(baseline_fname)

            # find the baseline file corresponding to the trial number of the data
            trial_num_str = name[name.find("_trial"): name.find("_button") + 1]
            baseline_fname = None
            for fname in all_baseline_fnames:
                if trial_num_str in fname:
                    baseline_fname = fname
                    break
            # get the baseline data
            baseline_filepath = os.path.join(baseline_dirpath, baseline_fname)
            baseline_data = self.__get_file_data(baseline_filepath)
            """

        # return baseline_data
        return baseline_dict

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

            # calculating average of all frequencies for each signal
            # stat_array = self.get_attribute_stats(file_lines)
            # stat_array = self.get_gesture_stats(file_lines)
            # return stat_array
            return file_lines

    @staticmethod
    def get_gesture_stats(data):

        mean = np.expand_dims(np.mean(data, axis=1), axis=1)
        std = np.expand_dims(np.std(data, axis=1), axis=1)
        sum = np.expand_dims(np.sum(data, axis=1), axis=1)

        linreg_ls = []
        x = np.arange(0, data.shape[1])
        for i in range(0, data.shape[0]):
            time_step = data[i, :]

            # linear regression fitting
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, time_step)
            linreg = [slope, intercept, r_value, p_value, std_err]
            linreg_ls.append(linreg)

        linreg_stats = np.array([np.array(xi) for xi in linreg_ls])

        stat_array = np.concatenate((mean, std, sum, linreg_stats), axis=1)
        return stat_array

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
        even_linreg_ls = []
        for i in range(0, odd_column_array.shape[0]):
            odd_time_step = odd_column_array[i, :]
            even_time_step = even_column_array[i, :]

            # linear regression fitting
            odd_slope, odd_intercept, odd_r_value, odd_p_value, odd_std_err = stats.linregress(x, odd_time_step)
            odd_linreg = [odd_slope, odd_intercept, odd_r_value, odd_p_value, odd_std_err]
            odd_linreg_ls.append(odd_linreg)

            even_slope, even_intercept, even_r_value, even_p_value, even_std_err = stats.linregress(x, even_time_step)
            even_linreg = [even_slope, even_intercept, even_r_value, even_p_value, even_std_err]
            even_linreg_ls.append(even_linreg)

            # plotting
            # plt.plot(x, odd_time_step, 'o', label='original data')
            # plt.plot(x, x*odd_slope + odd_intercept, label='fitted line')
            # plt.legend()
            # plt.show()

        odd_linreg_stats = np.array([np.array(xi) for xi in odd_linreg_ls])
        even_linreg_stats = np.array([np.array(xi) for xi in even_linreg_ls])
        # """

        stat_array = np.concatenate((odd_mean, odd_std, odd_sum, odd_linreg_stats,
                                     even_mean, even_std, even_sum, even_linreg_stats), axis=1)
        # stat_array = np.concatenate((odd_mean, even_mean, odd_std, even_std, odd_sum, even_sum,
        #                              odd_linreg_stats, even_linreg_stats), axis=1)
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
