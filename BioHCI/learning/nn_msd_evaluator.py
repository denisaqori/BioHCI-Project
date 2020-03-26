"""
Created: 2/9/20
© Denisa Qori McDonald 2020 All Rights Reserved
"""
import math
import random
from typing import List

import numpy as np

from BioHCI.data_processing.keypoint_description.ELD import ELD
from BioHCI.data_processing.keypoint_description.desc_type import DescType
from BioHCI.data_processing.keypoint_description.interval_descriptor import IntervalDescription
from BioHCI.data_processing.keypoint_description.sequence_length import SeqLen
from BioHCI.learning.evaluator import Evaluator


class NN_MSD_Evaluator(Evaluator):
    def __init__(self, msd_labels_dict, desc_type, seq_len, knitted_component, model_to_eval, criterion,
                 neural_network_def, parameters, summary_writer):

        super(NN_MSD_Evaluator, self).__init__(model_to_eval, criterion, neural_network_def, parameters, summary_writer)

        assert msd_labels_dict is not None
        self.__msd_train_labels_dict = msd_labels_dict
        self.__knitted_component = knitted_component
        self.__distance = ELD()
        self.__desc_type = desc_type
        self.__seq_len = seq_len

    @property
    def desc_type(self):
        return self.__desc_type

    @property
    def seq_len(self):
        return self.__seq_len

    @property
    def msd_train_labels_dict(self):
        return self.__msd_train_labels_dict

    @property
    def knitted_component(self):
        return self.__knitted_component

    @property
    def distance(self):
        return self.__distance

    def evaluate(self, val_data_loader, confusion, buttons_correct, fold_count, overall_rows_correct):
        # number of correct guesses
        row_correct = 0
        # button_correct = 0

        total = 0
        row_loss = 0

        # Go through the test dataset and record which are correctly guessed
        for step, (data_chunk_tensor, row_category_tensor, button_category_tensor) in enumerate(val_data_loader):

            # data_chunk_tensor has shape (batch_size x samples_per_step x num_attr)
            # category_tensor has shape (batch_size)
            # batch_size is passed as an argument to train_data_loader
            if self._parameters.classification:
                row_category_tensor = row_category_tensor.long()  # the loss function requires it
                button_category_tensor = button_category_tensor.long()  # the loss function requires it
            else:
                row_category_tensor = row_category_tensor.float()
                button_category_tensor = button_category_tensor.float()
            data_chunk_tensor = data_chunk_tensor.float()

            # getting the architectures guess for the category
            output, loss = self._evaluate_chunks_in_batch(data_chunk_tensor, row_category_tensor, self._model_to_eval)
            row_loss += loss

            # for every element of the batch
            for i in range(0, len(row_category_tensor)):
                total = total + 1
                # calculating true category

                # calculating predicted categories for the whole batch
                assert self._parameters.classification

                row_category_i = int(row_category_tensor[i])
                row_predicted_i = self._category_from_output(output[i])

                raw_sample = data_chunk_tensor[i, :, :].numpy()

                button_predicted_i = self.predict_button(raw_sample, row_predicted_i)
                button_category_i = int(button_category_tensor[i])

                # adding data to the matrix
                confusion[row_category_i][row_predicted_i] += 1

                fold_count += 1
                if row_category_i == row_predicted_i:
                    row_correct += 1
                    overall_rows_correct += 1
                    if button_predicted_i == button_category_i:
                        buttons_correct += 1

        row_accuracy = row_correct / total
        return row_loss, row_accuracy

    def predict_button(self, raw_sample: np.ndarray, row_number: int) -> int:
        button_ls = self.get_buttons_of_row(row_number)
        avg_dist_ls = []
        for button in button_ls:
            samples = self.get_button_samples(button)
            avg_dist = self.get_avg_dist(raw_sample, samples)
            avg_dist_ls.append(avg_dist)

        # the index of minimum distance
        min_idx = avg_dist_ls.index(min(avg_dist_ls))
        closest_button = button_ls[min_idx]
        return closest_button

    def get_buttons_of_row(self, row_num: int) -> List[int]:
        num_buttons_per_row = math.floor(self.knitted_component.num_buttons / self.knitted_component.num_rows)

        button_ls = []
        for i in range(0, num_buttons_per_row):
            button = num_buttons_per_row * row_num + i
            button_ls.append(button)
        return button_ls

    def get_button_samples(self, label: int, num_rand: int = 5) -> List[np.ndarray]:
        all_samples = self.msd_train_labels_dict[label]
        a = list(range(0, len(all_samples)))
        random_idxs = random.choices(a, k=num_rand)
        # random_idxs = random.sample(a, k=num_rand)
        random_samples = [all_samples[i] for i in random_idxs]

        return random_samples

    def get_avg_dist(self, raw_inst: np.ndarray, sample_ls: List[np.ndarray]) -> float:
        msd_desc = IntervalDescription(raw_inst, DescType.MSD).descriptors[0]

        distance_ls = []
        for rand_sample in sample_ls:
            # msd_desc, rand_sample = self.equalize_desc_size(msd_desc, rand_sample)
            d = self.distance.compute_distance(msd_desc, rand_sample)
            distance_ls.append(d)

        avg_dist = sum(distance_ls) / len(distance_ls)
        return avg_dist

    def equalize_desc_size(self, desc1, desc2):
        if desc1.shape[0] == desc2.shape[0]:
            return desc1, desc2
        elif desc1.shape[0] > desc2.shape[0]:
            smaller = desc2
            bigger = desc1
        else:
            smaller = desc1
            bigger = desc2

        num_rows_to_add = bigger.shape[0] - smaller.shape[0]
        if self.seq_len == SeqLen.ExtendEdge:
            smaller = np.pad(smaller, [(0, num_rows_to_add), (0, 0)], mode='edge')

        elif self.seq_len == SeqLen.ZeroPad:
            smaller = np.pad(smaller, [(0, num_rows_to_add), (0, 0)], mode='constant', constant_values=0)
        else:
            print("Sequence length type undefined. Returning original descriptors.")

        return smaller, bigger



