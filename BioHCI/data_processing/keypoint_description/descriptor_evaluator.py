"""
Created: 3/27/19
© Denisa Qori McDonald 2019 All Rights Reserved
"""
import multiprocessing
import ctypes
import time

import numpy as np
from scipy import stats
from BioHCI.data.data_constructor import DataConstructor
from BioHCI.data_processing.keypoint_description.ELD import ELD
from BioHCI.data_processing.keypoint_description.desc_type import DescType
from BioHCI.data_processing.keypoint_description.descriptor_computer import DescriptorComputer
from BioHCI.helpers.study_config import StudyConfig
import pickle
from BioHCI.helpers import utilities as utils
import seaborn as sns
import matplotlib.pyplot as plt
import os
import BioHCI.helpers.type_aliases as types
from typing import List, Tuple, Optional
from os.path import join
import logging
from datetime import datetime
from BioHCI.data_processing.keypoint_description.sequence_length import SeqLen


class DescriptorEvaluator:
    def __init__(self, descriptor_computer: DescriptorComputer, dataset_descriptors: types.subj_dataset,
                 all_dataset_categories: List[str], heatmap_global: np.ndarray,
                 heatmap_alt_lables: Optional[List[str]] = None) -> None:

        # always use heatmap_global, not self.__heatmap in the parallelized section when writing to the array. Each
        # process has its own copy of a class and its variables, so self.__heatmap would not reflect all changes if
        # it was written to by several processes. For each process, its self.__heatmap is set to the same memory
        # location as heatmap_global.

        print(f"\nPerforming descriptor dataset evaluation...\n")

        self.__distance = ELD()
        self.__heatmap = heatmap_global
        self.descriptor_computer = descriptor_computer

        # if there is no root directory for dataset descriptors, create it
        eval_subdir = parameters.study_name + "/descriptor_evaluation"
        self.__dataset_eval_dir = utils.create_dir(join(utils.get_root_path("saved_objects"), eval_subdir))

        # create the full path of the dataset evaluation object
        self.__eval_obj_path = join(self.dataset_eval_dir, self.dataset_eval_name)

        # create a directory under Results to save the resulting heatmap figure and result logs.
        self.__results_eval_dir = utils.create_dir(join(utils.get_root_path("Results"), eval_subdir))

        # Removes any files that contain the string "_test" in the dataset evaluation directory under saved_objects,
        # as well as any saved heatmaps or statistic text files with that name under Results directory.
        utils.cleanup(self.dataset_eval_dir, "_test")
        utils.cleanup(self.results_eval_dir, "_test")

        self.__num_processes = multiprocessing.cpu_count() * 2
        self.compute_heatmap(dataset_descriptors, all_dataset_categories, heatmap_alt_lables)

        # defining the logger before the multiprocessing task causes a "cannot pickle RLock error" since
        # the logger holds a lock to the file.
        self.__result_logger = self.define_result_logger()
        print("")

    def define_result_logger(self) -> logging.Logger:
        """
        Creates a custom logger to write the results of the dataset evaluation on the console and file.

        Returns:
            logger(Logging.logger): the logger used to report statistical results.

        """
        # Create a custom logger
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.DEBUG)

        # Create handlers
        c_handler = logging.StreamHandler()
        c_handler.setLevel(logging.INFO)

        results_log_path = join(self.results_eval_dir, self.dataset_eval_name + "_statistics.txt")
        f_handler = logging.FileHandler(filename=results_log_path)
        f_handler.setLevel(logging.DEBUG)

        # Add handlers to the logger
        logger.addHandler(c_handler)
        logger.addHandler(f_handler)

        print(f"Logging into file: {results_log_path}")
        return logger

    @property
    def distance(self):
        return self.__distance

    @property
    def num_processes(self) -> int:
        return self.__num_processes

    @property
    def result_logger(self) -> logging.Logger:
        return self.__result_logger

    @property
    def dataset_eval_name(self) -> str:
        return self.descriptor_computer.dataset_desc_name + "_heatmap"

    @property
    def dataset_eval_dir(self) -> str:
        return self.__dataset_eval_dir

    @property
    def eval_obj_path(self) -> str:
        return self.__eval_obj_path

    @property
    def results_eval_dir(self) -> str:
        return self.__results_eval_dir

    # @property
    # def dataset_descriptors(self) -> types.subj_dataset:
    #     return self.descriptor_computer.dataset_descriptors

    @property
    def heatmap(self) -> Optional[np.ndarray]:
        return self.__heatmap

    def compute_heatmap(self, dataset_descriptors: types.subj_dataset, all_dataset_categories: List[str],
                        labels: List [str] = None) -> None:
        """
        Computes pairwise distances of all tensors in the descriptors (internal to the class) and accumulates the sum
        of the distances among pairs of all categories into a heatmap.

        Args:
            dataset_descriptors: descriptors of the dataset to be evaluated
            all_dataset_categories (list): a list of all categories of the subject dataset.

        Returns: None

        """
        print("\nComputing evaluation matrix...")

        if not os.path.exists(self.get_heatmap_obj_path()):
            start_time = time.time()
            for subj_name, subj in dataset_descriptors.items():
                subj_data = subj.data
                subj_cat = subj.categories
                cat_map = utils.map_categories(all_dataset_categories)
                subj_int_cat = utils.convert_categories(cat_map, subj_cat)
                print(f"Category to Integer Mapping: {cat_map}")

                tuple_list = []
                for i in range(0, len(subj_data)):
                    for j in range(0, len(subj_data)):
                        keypress1 = subj_data[i]
                        cat1 = subj_int_cat[i]

                        keypress2 = subj_data[j]
                        cat2 = subj_int_cat[j]

                        tuple_list.append((keypress1, cat1, keypress2, cat2))

                print(f"Id of heatmap as seen by main: {hex(id(heatmap_global))}")
                print(f"Id of heatmap as seen by compute_heatmap() method: {hex(id(self.heatmap))}")
                print(f"Total number of tensors to compare is {len(tuple_list)}")

                with multiprocessing.Pool(processes=self.num_processes) as pool:
                    pool.map(self.compute_distance_parallelized, tuple_list)
                pool.close()
                pool.join()
            duration_with_pool = utils.time_since(start_time)

            print(f"\nComputed dataset evaluation heatmap for {len(subject_dataset)} subject(s), "
                  f"using {self.num_processes} processes, for a duration of {duration_with_pool}\n")

            self.save(self.heatmap, ".pkl")
        else:
            print("Opening existing heatmap...")
            with (open(self.get_heatmap_obj_path(), "rb")) as openfile:
                self.__heatmap = pickle.load(openfile)

        if self.heatmap is not None:
            plt.figure(figsize=(18, 15))
            sns.set(font_scale=2.5)
            if labels:
                heatmap_fig = sns.heatmap(self.heatmap, xticklabels=labels, yticklabels=labels)
                plt.yticks(rotation=45)
                plt.xticks(rotation=45)
            else:
                heatmap_fig = sns.heatmap(self.heatmap, xticklabels=1, yticklabels=1)
            self.save(heatmap_fig, ".png")

        print(f"\nEnd of descriptor evaluator {self.dataset_eval_name}!\n")

    def compute_distance_seq(self, all_dataset_categories: List[str]):

        for subj_name, subj in self.dataset_descriptors.items():
            subj_data = subj.data
            subj_cat = subj.categories
            cat_map = utils.map_categories(all_dataset_categories)
            subj_int_cat = utils.convert_categories(cat_map, subj_cat)

            # for testing purposes
            l1 = []
            l2 = []

            tuple_list = []
            for i in range(0, len(subj_data)):
                for j in range(0, len(subj_data)):
                    keypress1 = subj_data[i]
                    cat1 = subj_int_cat[i]

                    keypress2 = subj_data[j]
                    cat2 = subj_int_cat[j]

                    # for testing purposes
                    lev_dist_1 = self.distance.compute_distance(keypress1, keypress2)
                    l1.append(lev_dist_1)
                    lev_dist_2 = self.distance.compute_distance(keypress2, keypress1)
                    l2.append(lev_dist_2)

            lists_same = (l1 == l2)
            zero_in_l1 = (0 in l1)
            print(f"Are lists the same: {lists_same}")
            print(f"Does l1 contain a 0? {zero_in_l1}")
            print()

    def compute_distance_parallelized(self, args):
        """
        Computes the distance between two tensors. This is the target function of a multiprocessing pool. Adds the
        computed distance to the appropriate heatmap coordinates, which is defined by the categories each of the two
        tensors belongs to. The heatmap object is shared among processes. !!!!! Make sure to edit heatmap_global,
        not self.heatmap or self.__heatmap.

        Args:
            args (tuple): a tuple containing the first tensor, its category, the second tensor, its category

        Returns: None

        """
        keypress1, cat1, keypress2, cat2 = args

        lev_dist = self.distance.compute_distance(keypress1, keypress2)

        heatmap_global[cat1, cat2] = heatmap_global[cat1, cat2] + lev_dist
        with counter.get_lock():
            counter.value += 1
            if counter.value % 100 == 0:
                print(f"{counter.value}: Process {multiprocessing.current_process()}")

    def save(self, obj, ext: str, extra_name: str = "") -> None:
        """
        Saves an object to a file (pickles or saves a figure to png).

        Args:
            obj: object to save
            ext (str): extension of the file which will be created to save the object
            extra_name (str): any additional string to be added to the descriptor evaluator's name which will be used
                to name the file.

        Returns: None

        """

        if ext == ".pkl":
            path = join(self.dataset_eval_dir, self.dataset_eval_name + extra_name + ext)
            with open(path, 'wb') as f:
                pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
            print(f"Saved file: {path}")

        elif ext == ".png":
            path = join(self.results_eval_dir, self.dataset_eval_name + extra_name + ext)
            obj.figure.savefig(path)
            print(f"Saved file: {path}")

            plt.show()
            plt.close("all")

        else:
            print("Invalid extension. Object not saved!")

    def get_heatmap_obj_path(self) -> str:
        """
        Returns: path of the generated heatmap object.
        """
        path = join(self.dataset_eval_dir, self.dataset_eval_name + ".pkl")
        return path

    @staticmethod
    def get_category_distance_stats(heatmap_matrix: np.ndarray) -> Tuple[float, float, float, float, float, float,
                                                                         float, float, float, float]:
        """
        Calculates statistics on the heatmap object passed, regarding class similarities and differences.

        Args:
            heatmap_matrix (ndarray): a matrix containing the distances between classes among all samples

        Returns:
            avg_same (float): average distance among same-class tensors
            avg_diff (float): average distance among tensors from different classes
            std_same (float): standard deviation of the distance among same-class tensors
            std_diff (float): standard deviation of the distance among tensors from different classes
            cv_same (float): coefficient of variation among same-class tensors
            cv_diff (float): coefficient of variation among tensors of different classes

        """
        same_list = []
        diff_list = []
        for i in range(0, heatmap_matrix.shape[0]):
            for j in range(0, heatmap_matrix.shape[1]):
                if i == j:
                    same_list.append(heatmap_matrix[i, j])
                else:
                    diff_list.append(heatmap_matrix[i, j])

        avg_same = np.mean(np.asarray(same_list))
        avg_diff = np.mean(np.asarray(diff_list))

        std_same = np.std(np.asarray(same_list))
        std_diff = np.std(np.asarray(diff_list))

        cv_same = std_same / avg_same
        cv_diff = std_diff / avg_diff

        # same calculation as z_score in this case
        z_score, p_val_z = stats.ttest_ind(same_list, diff_list)
        f_stat, p_val_f = stats.f_oneway(same_list, diff_list)

        return avg_same, avg_diff, std_same, std_diff, cv_same, cv_diff, z_score, p_val_z, f_stat, p_val_f

    def log_statistics(self) -> None:
        """
        Calls get_category_distance_stats() to obtain distance statistics among tensors, and logs the results in a
        text file named after the descriptor evaluator, together with the fully computed heatmap.

        Returns: None

        """
        now = datetime.now()
        self.result_logger.info(f"\nTime: {now:%A, %d. %B %Y %I: %M %p}")
        self.result_logger.info(f"{self.dataset_eval_name}\n")

        self.result_logger.debug(f"Heatmap: \n")
        self.result_logger.debug(f"{self.heatmap}\n\n")

        avg_same, avg_diff, std_same, std_diff, cv_same, cv_diff, z_score, p_val_z, \
        f_stat, p_val_f = self.get_category_distance_stats(self.heatmap)

        ratio_same_diff = avg_same / avg_diff

        self.result_logger.info(
            f"Average distance among same-class tensors is                                  {avg_same:.3f}")
        self.result_logger.info(
            f"Average distance among tensors of different classes is                        {avg_diff:.3f}")
        self.result_logger.info(
            f"Standard deviation of the distance among same-class tensors is                {std_same:.3f}")
        self.result_logger.info(
            f"Standard deviation of the distance among tensors of different classes is      {std_diff:.3f}")
        self.result_logger.info(
            f"Coefficient of variation among same-class tensors is                          {cv_same:.3f}")
        self.result_logger.info(
            f"Coefficient of variation among tensors of different classes is                {cv_diff:.3f}")
        self.result_logger.info(
            f"Ratio of same-class average distance to different class average distance is   {ratio_same_diff:.3f}")
        self.result_logger.info(
            f"The z-score of same-class distances and different-class distances is          {z_score:.3f}")
        self.result_logger.info(
            f"The p-value determined by the z-test is                                       {p_val_z:.3f}")
        self.result_logger.info(
            f"The f-statistic of same-class distances and different-class distances is      {f_stat:.3f}")
        self.result_logger.info(
            f"The p-value determined by the f-test is                                       {p_val_f:.3f}\n")
        self.result_logger.info(
            f"**********************************************************************************************************************")
        self.result_logger.info(
            f"**********************************************************************************************************************\n")
        # close and detach all handlers
        handlers = self.result_logger.handlers[:]
        for handler in handlers:
            handler.close()
            self.result_logger.removeHandler(handler)

        logging.shutdown()

    def generate_heatmap_fig_from_obj(self, heatmap: np.ndarray) -> None:
        """
        Given a heatmap matrix, produces its heatmap figure with the same name, except for the extension (.pkl vs .png)

        Args:
            heatmap(ndarray): a matrix containing the distances between classes among all samples

        """
        plt.figure(figsize=(14, 10))
        sns.set(font_scale=2)
        # heatmap_fig = sns.heatmap(heatmap, xticklabels=1, yticklabels=1)
        heatmap_fig = sns.heatmap(heatmap)

        self.save(heatmap_fig, ".pdf")

    def generate_heatmap_fig_from_obj_name(self, heatmap_name: str) -> None:
        """
        Given a heatmap name, produces its heatmap figure.

        Args:
            heatmap_name: the name of the pickled heatmap object to convert into a figure

        """
        assert heatmap_name.endswith(".pkl")
        path = join(self.dataset_eval_dir, heatmap_name)
        if os.path.exists(path):
            with (open(path, "rb")) as openfile:
                heatmap = pickle.load(openfile)
                self.generate_heatmap_fig_from_obj(heatmap)


if __name__ == "__main__":
    np.set_printoptions(threshold=10000, linewidth=100000, precision=1)
    config_dir = "config_files"
    config = StudyConfig(config_dir)

    # create a template of a configuration file with all the fields initialized to None
    config.create_config_file_template()
    # parameters = config.populate_study_parameters("CTS_5taps_per_button.toml")
    # parameters = config.populate_study_parameters("CTS_volume_swipe.toml")
    parameters = config.populate_study_parameters("CTS_4Electrodes.toml")

    # generating the data from files
    data = DataConstructor(parameters)
    subject_dataset = data.cv_subj_dataset

    # get all the categories of the dataset
    all_dataset_categories = data.get_all_dataset_categories()

    # determine the shape of the array
    heatmap_shape = (len(set(all_dataset_categories)), len(set(all_dataset_categories)))

    """
    # generate statistics for all descriptors
    for desc_type in DescType:
        for norm_bool in [True, False]:
            print(f"Descriptor computing and evaluation on {desc_type} with l2-normalization set to {norm_bool}.")

            # create a counter, lock to give to shared array
            counter = multiprocessing.Value('i', 0)
            lock = multiprocessing.Lock()
            # create shared numpy array
            shared_array_base = multiprocessing.Array(ctypes.c_double, heatmap_shape[0] * heatmap_shape[1], lock=lock)
            shared_array = np.ctypeslib.as_array(shared_array_base.get_obj())
            heatmap_global = shared_array.reshape(heatmap_shape)

            # create descriptor computer
            desc_computer = DescriptorComputer(desc_type, subject_dataset, parameters, normalize=norm_bool)
            # evaluate distances between tensors and compute statistics on them
            desc_eval = DescriptorEvaluator(desc_computer, all_dataset_categories, heatmap_global)
            desc_eval.log_statistics()
    """

    counter = multiprocessing.Value('i', 0)
    lock = multiprocessing.Lock()
    # create shared numpy array
    shared_array_base = multiprocessing.Array(ctypes.c_double, heatmap_shape[0] * heatmap_shape[1], lock=lock)
    shared_array = np.ctypeslib.as_array(shared_array_base.get_obj())
    heatmap_global = shared_array.reshape(heatmap_shape)

    # create descriptor computer
    desc_computer = DescriptorComputer(DescType.RawData, subject_dataset, parameters, seq_len=SeqLen.Existing,
                                       extra_name="_chi2021_final")
    descriptors = desc_computer.produce_dataset_descriptors(subject_dataset)

    heatmap_alt_lables = ["Double", "Single", "Half", "Random"]
    # evaluate distances between tensors and compute statistics on them
    desc_eval = DescriptorEvaluator(desc_computer, descriptors, all_dataset_categories, heatmap_global,
                                    heatmap_alt_lables)
    # desc_eval.generate_heatmap_fig_from_obj_name(desc_eval.dataset_eval_name + ".pkl")
    desc_eval.log_statistics()

    print("")
    # """
