"""
Created: 3/27/19
© Denisa Qori McDonald 2019 All Rights Reserved
"""
from concurrent.futures import ThreadPoolExecutor

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
import threading
import concurrent.futures


class DescriptorEvaluator:
    def __init__(self, descriptor_computer, subject_dataset):
        if descriptor_computer.dataset_desc_path is not None and os.path.exists(
                descriptor_computer.dataset_desc_path):
            print("Loading dataset descriptors from: ", descriptor_computer.dataset_desc_path)
            self.__dataset_descriptors_dict = self.load_descriptors(descriptor_computer.dataset_desc_path)
        else:
            print("Producing dataset descriptors...")
            self.__dataset_descriptors_dict = descriptor_computer.produce_dataset_descriptors(subject_dataset)
        self.descriptor_computer = descriptor_computer

        dataset_eval_path = '/home/denisa/GitHub/BioHCI ' \
                            'Project/BioHCI/data_processing/keypoint_description/dataset_evals'
        self.dataset_eval_dir = utils.create_dir(dataset_eval_path)

    @property
    def dataset_descriptors_dict(self):
        assert self.dataset_descriptors_dict is not None
        return self.__dataset_descriptors_dict

    @staticmethod
    def load_descriptors(dataset_desc_path):
        with open(dataset_desc_path, "rb") as input_file:
            dataset_desc = pickle.load(input_file)
        return dataset_desc

    def compute_heatmap(self, all_dataset_categories):

        heatmap = None

        if not os.path.exists(self.get_matrix_full_name()):
            for subj_name, subj in self.__dataset_descriptors_dict.items():
                subj_data = subj.data
                subj_cat = subj.categories
                subj_int_cat = utils.convert_categories(all_dataset_categories, subj_cat)

                heatmap = np.zeros((len(set(subj_int_cat)), len(set(subj_int_cat))))

                num = 0
                for i in range(0, len(subj_data) - 1):
                    for j in range(0, len(subj_data) - 1):
                        keypress1 = subj_data[i]
                        cat1 = subj_int_cat[i]

                        keypress2 = subj_data[j + 1]
                        cat2 = subj_int_cat[j]
                        # print("cat 1: ", cat1, "cat 2: ", cat2)
                        print("Number of levenshtine dist computed: ", num)

                        # lev_dist = self.levenshtein_distance(keypress1, keypress2)
                        lev_dist = self.real_levenshtein_distance(keypress1, keypress2)

                        with threading.Lock():
                            # the next two lines need to be locked
                            heatmap[cat1, cat2] = heatmap[cat1, cat2] + lev_dist
                            num = num + 1

                self.save_obj(heatmap, ".pkl", "_matrix")

        else:
            with (open(self.get_matrix_full_name(), "rb")) as openfile:
                heatmap = pickle.load(openfile)

        if heatmap is not None:
            plt.figure(figsize=(14, 10))
            sns.set(font_scale=1.4)
            heatmap_fig = sns.heatmap(heatmap, xticklabels=5, yticklabels=5)
            self.save_obj(heatmap_fig, ".png", "_heatmap")

        return heatmap

    @staticmethod
    def levenshtein_distance(keypress1, keypress2):
        lev_matrix = np.zeros((keypress1.shape[0], keypress2.shape[0]))
        for i in range(1, keypress1.shape[0]):
            for j in range(1, keypress2.shape[0]):
                kpress1_current_keypoint = keypress1[i, :]
                kpress2_current_keypoint = keypress2[j, :]
                current_dist = np.linalg.norm(kpress1_current_keypoint - kpress2_current_keypoint)

                diag = lev_matrix[i - 1, j - 1]
                left = lev_matrix[i - 1, j]
                up = lev_matrix[i, j - 1]

                lev_matrix[i, j] = min(diag, left, up) + current_dist

        # return the element of the last diagonal
        minimal_cost = lev_matrix[keypress1.shape[0] - 1, keypress2.shape[0] - 1]
        return minimal_cost

    @staticmethod
    def real_levenshtein_distance(keypress1, keypress2):
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

        # return the element of the last diagonal
        minimal_cost = lev_matrix[keypress1.shape[0] - 1, keypress2.shape[0] - 1]
        return minimal_cost

    def save_obj(self, obj, ext, extra_name=""):
        dataset_eval_path = os.path.abspath(os.path.join(self.dataset_eval_dir,
                                                         self.descriptor_computer.parameters.study_name +
                                                         str(
                                                             self.descriptor_computer.desc_type))
                                            + self.descriptor_computer.dataset_desc_name + extra_name + ext)
        if ext == ".pkl":
            with open(dataset_eval_path, 'wb') as f:
                pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
        elif ext == ".png":
            obj.figure.savefig(dataset_eval_path)
            plt.show()
            plt.close("all")
        else:
            print("Invalid extension. Object not saved!")

    def get_matrix_full_name(self):
        matrix_path = os.path.abspath(os.path.join(self.dataset_eval_dir,
                                                   self.descriptor_computer.parameters.study_name + "_desc_type_" + str(
                                                       self.descriptor_computer.desc_type))
                                      + self.descriptor_computer.dataset_desc_name + "_matrix.pkl")
        return matrix_path

    @staticmethod
    def get_avg_category_distance(heatmap_matrix):

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


if __name__ == "__main__":
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

        # MSBSD compution - normalized
    descriptor_2_computer_norm = DescriptorComputer(DescType.MSBSD, parameters, normalize=True)
    descriptor_2_eval_norm = DescriptorEvaluator(descriptor_2_computer_norm, subject_dataset)

    executor = ThreadPoolExecutor(max_workers=64)
    heatmap_matrix_2_norm = executor.submit(descriptor_2_eval_norm.compute_heatmap(data.get_all_dataset_categories()))

    # heatmap_matrix_2_norm = descriptor_2_eval_norm.compute_heatmap(data.get_all_dataset_categories())
    avg_same_2_norm, avg_diff_2_norm, std_same_2_norm, std_diff_2_norm, cv_same_2_norm, cv_diff_2_norm = \
        descriptor_2_eval_norm.get_avg_category_distance(heatmap_matrix_2_norm)
    ratio_2_norm = avg_same_2_norm / avg_diff_2_norm

    f = open("desc_eval_msbsd_norm.txt", "w")
    f.write("avg_same_norm: %f\r\n" %(avg_same_2_norm))
    f.write("avg_diff_norm: %f\r\n" %(avg_diff_2_norm))
    f.write("std_same_norm: %f\r\n" %(std_same_2_norm))
    f.write("std_diff_norm: %f\r\n" %(std_diff_2_norm))
    f.write("cv_same_norm: %f\r\n" %(cv_same_2_norm))
    f.write("cv_diff_norm: %f\r\n" %(cv_diff_2_norm))
    f.write("ratio_norm: %f\r\n" %(ratio_2_norm))
    f.close()

    print("")
