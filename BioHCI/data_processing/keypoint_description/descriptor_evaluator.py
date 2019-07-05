"""
Created: 3/27/19
© Denisa Qori McDonald 2019 All Rights Reserved
"""
import multiprocessing
import ctypes
import time

import numpy as np
from BioHCI.data.data_constructor import DataConstructor
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


shared_array_base = multiprocessing.Array(ctypes.c_double, 36*36, lock=True)
shared_array = np.ctypeslib.as_array(shared_array_base.get_obj())
heatmap = shared_array.reshape((36, 36))

# TODO: implement logging
class DescriptorEvaluator:
    def __init__(self, descriptor_computer: DescriptorComputer, all_dataset_categories: List[str]) -> None:

        self.__heatmap = None
        self.descriptor_computer = descriptor_computer

        dataset_eval_path = utils.get_root_path("dataset_eval")
        self.dataset_eval_dir = utils.create_dir(dataset_eval_path)
        self.cleanup()

        self.compute_heatmap(all_dataset_categories)

    @property
    def dataset_eval_name(self) -> str:
        return self.descriptor_computer.dataset_desc_name + "_heatmap"

    @property
    def dataset_descriptors(self) -> types.subj_dataset:
        return self.descriptor_computer.dataset_descriptors

    @property
    def heatmap(self) -> Optional[np.ndarray]:
        return self.__heatmap

    def compute_heatmap(self, all_dataset_categories: List[str]) -> None:

        if not os.path.exists(self.get_heatmap_obj_path()):
            for subj_name, subj in self.dataset_descriptors.items():
                subj_data = subj.data
                subj_cat = subj.categories
                subj_int_cat = utils.convert_categories(all_dataset_categories, subj_cat)

                lock = multiprocessing.Lock()
                heatmap_shape = (len(set(subj_int_cat)), len(set(subj_int_cat)))

                # shared_array_base = multiprocessing.Array(ctypes.c_double, heatmap_shape[0]*heatmap_shape[1], lock=lock)
                # shared_array = np.ctypeslib.as_array(shared_array_base.get_obj())
                # self.__heatmap = shared_array.reshape(heatmap_shape)

                tuple_list = []
                # for i in range(0, len(subj_data) - 1):
                #     for j in range(0, len(subj_data) - 1):
                for i in range(0, 7):
                    for j in range(0, 7):
                        keypress1 = subj_data[i]
                        cat1 = subj_int_cat[i]

                        keypress2 = subj_data[j]
                        cat2 = subj_int_cat[j]

                        tuple_list.append((keypress1, cat1, keypress2, cat2))

                num_processes = multiprocessing.cpu_count()
                start_time = time.time()
                print("Starting...")
                print("Id of heatmap as seen by main: ", hex(id(heatmap)))
                with multiprocessing.Pool(processes=num_processes) as pool:
                    pool.map(self.compute_distance_parallelized, tuple_list)
                duration_with_pool = utils.time_since(start_time)
                # print(heatmap)

                print("Computed dataset descriptors for subject {}, using {} processes, for a duration of {}".format(
                    subj_name, num_processes, duration_with_pool))
                self.save_obj(self.__heatmap, ".pkl")
        else:
            print("Opening existing heatmap...")
            with (open(self.get_heatmap_obj_path(), "rb")) as openfile:
                self.__heatmap = pickle.load(openfile)

        if self.__heatmap is not None:
            plt.figure(figsize=(14, 10))
            sns.set(font_scale=1.4)
            heatmap_fig = sns.heatmap(self.__heatmap, xticklabels=5, yticklabels=5)
            self.save_obj(heatmap_fig, ".png")

        print("End of descriptor evaluator!")

    def compute_distance_parallelized(self, args):
        keypress1, cat1, keypress2, cat2 = args

        lev_dist = self.real_levenshtein_distance(keypress1, keypress2)
        # self.__heatmap[cat1, cat2] = self.__heatmap[cat1, cat2] + lev_dist

        print (multiprocessing.current_process())
        print(hex(id(heatmap)))
        heatmap[cat1, cat2] = heatmap[cat1, cat2] + lev_dist

        print("[{}, {}] - {}".format(cat1, cat2, heatmap[cat1,cat2]))
        #print("")

    @staticmethod
    def real_levenshtein_distance(keypress1: np.ndarray, keypress2: np.ndarray) -> float:
        lev_matrix = np.zeros((keypress1.shape[0], keypress2.shape[0]))
        for i in range(1, keypress1.shape[0]):
            for j in range(1, keypress2.shape[0]):

                k1_i = keypress1[i, :]
                k2_j = keypress2[j, :]

                k1_i_norm = np.linalg.norm(k1_i)
                k2_j_norm = np.linalg.norm(k2_j)
                diff_norm = np.linalg.norm(k1_i - k2_j)

                if min(i, j) == 0:
                    lev_matrix[i, j] = max(k1_i_norm, k2_j_norm)

                else:
                    left = lev_matrix[i - 1, j]
                    min_clause_1 = left + k1_i_norm

                    up = lev_matrix[i, j - 1]
                    min_clause_2 = up + k2_j_norm

                    diag = lev_matrix[i - 1, j - 1]
                    min_clause_3 = diag + diff_norm

                    lev_matrix[i, j] = min(min_clause_1, min_clause_2, min_clause_3)

        # return the last element of the diagonal
        minimal_cost = lev_matrix[keypress1.shape[0] - 1, keypress2.shape[0] - 1]
        return minimal_cost

    def save_obj(self, obj, ext: str, extra_name: str = "") -> None:
        path = join(self.dataset_eval_dir, self.dataset_eval_name + extra_name + ext)

        if ext == ".pkl":
            with open(path, 'wb') as f:
                pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
        elif ext == ".png":
            obj.figure.savefig(path)
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
    def get_category_distance_stats(heatmap_matrix: np.ndarray) -> Tuple[float, float, float, float, float, float]:
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
        return avg_same, avg_diff, std_same, std_diff, cv_same, cv_diff

    def generate_heatmap_fig_from_obj(self, heatmap: np.ndarray) -> None:
        """
        Given a heatmap matrix, produces its heatmap figure with the same name, except for the extension (.pkl vs .png)

        Args:
            heatmap(ndarray): a matrix containing the distances between classes among all samples

        """
        plt.figure(figsize=(14, 10))
        sns.set(font_scale=1.4)
        heatmap_fig = sns.heatmap(heatmap, xticklabels=5, yticklabels=5)
        self.save_obj(heatmap_fig, ".png")

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

    def cleanup(self):
        for filename in os.listdir(self.dataset_eval_dir):
            if "_test" in filename:
                full_path_to_remove = join(self.dataset_eval_dir, filename)

                print("Deleting file {}".format(full_path_to_remove))
                os.remove(full_path_to_remove)

if __name__ == "__main__":
    np.set_printoptions(threshold=10000, linewidth=100000, precision=8)
    config_dir = "config_files"
    config = StudyConfig(config_dir)

    # create a template of a configuration file with all the fields initialized to None
    config.create_config_file_template()
    parameters = config.populate_study_parameters("CTS_5taps_per_button.toml")

    # generating the data from files
    data = DataConstructor(parameters)
    subject_dataset = data.get_subject_dataset()

    """
    # JUSD compution - unnormalized
    descriptor_1_computer = DescriptorComputer(DescType.JUSD, parameters, normalize=False)
    descriptor_1_eval = DescriptorEvaluator(descriptor_1_computer, subject_dataset)
    heatmap_matrix_1 = descriptor_1_eval.compute_heatmap(data.get_all_dataset_categories())
    avg_same_1, avg_diff_1, std_same_1, std_diff_1, cv_same_1, cv_diff_1 = \
        descriptor_1_eval.get_avg_category_distance(heatmap_matrix_1)
    ratio_1 = avg_same_1 / avg_diff_1
        print (multiprocessing.current_process())
    # MSBSD compution - unnormalized
    descriptor_2_computer = DescriptorComputer(DescType.MSBSD, parameters, normalize=False)
    descriptor_2_eval = DescriptorEvaluator(descriptor_2_computer, subject_dataset)
    heatmap_matrix_2 = descriptor_2_eval.compute_heatmap(data.get_all_dataset_categories())
    avg_same_2, avg_diff_2, std_same_2, std_diff_2, cv_same_2, cv_diff_2 = \
        descriptor_2_eval.get_avg_category_distance(heatmap_matrix_2)
    ratio_2 = avg_same_2 / avg_diff_2

    # JUSD compution - normalized
    descriptor_1_computer_norm = DescriptorComputer(DescType.JUSD, parameters, normalize=True)
    descriptor_1_eval_norm = DescriptorEvaluator(descriptor_1_computer_norm, subject_dataset)
    heatmap_matrix_1_norm = descriptor_1_eval_norm.compute_heatmap(data.get_all_dataset_categories())
    avg_same_1_norm, avg_diff_1_norm, std_same_1_norm, std_diff_1_norm, cv_same_1_norm, cv_diff_1_norm = \
        descriptor_1_eval_norm.get_avg_category_distance(heatmap_matrix_1_norm)
    ratio_1_norm = avg_same_1_norm / avg_diff_1_norm
    """

    # MSBSD compution - unnormalized
    msbsd_computer_norm = DescriptorComputer(DescType.JUSD, subject_dataset, parameters, normalize=False,
                                             extra_name="_test_2")
    msbsd_eval_norm = DescriptorEvaluator(msbsd_computer_norm, data.get_all_dataset_categories())
    print(heatmap)

    # MSBSD compution - normalized
    # msbsd_computer_norm = DescriptorComputer(DescType.MSBSD, subject_dataset, parameters, normalize=True)
    # msbsd_eval_norm = DescriptorEvaluator(msbsd_computer_norm, data.get_all_dataset_categories())
    #
    # statistics = msbsd_eval_norm.get_category_distance_stats(msbsd_eval_norm.heatmap)
    # ratio_2_norm = statistics[0] / statistics[1]
    #
    # f = open("desc_eval_msbsd_norm.txt", "w")
    # f.write("avg_same_norm: %f\r\n" % statistics[0])
    # f.write("avg_diff_norm: %f\r\n" % statistics[1])
    # f.write("std_same_norm: %f\r\n" % statistics[2])
    # f.write("std_diff_norm: %f\r\n" % statistics[3])
    # f.write("cv_same_norm: %f\r\n" % statistics[4])
    # f.write("cv_diff_norm: %f\r\n" % statistics[5])
    # f.write("ratio_norm: %f\r\n" % ratio_2_norm)
    # f.close()
    #print(msbsd_eval_norm.heatmap)
    print("")
