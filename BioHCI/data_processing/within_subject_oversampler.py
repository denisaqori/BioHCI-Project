from BioHCI.data_processing.category_balancer import CategoryBalancer
import numpy as np
from copy import copy
import BioHCI.helpers.type_aliases as types
import BioHCI.helpers.utilities as utils
from typing import List


class WithinSubjectOversampler(CategoryBalancer):

    def balance(self, subj_dict: types.subj_dataset) -> types.subj_dataset:
        balanced = self.balance_flat_categories(subj_dict)

        # cat_balanced = self.balance_categories(compacted_subj_dict)
        return balanced

    def balance_flat_categories(self, subject_feature_dataset: types.subj_dataset) -> types.subj_dataset:
        """
        Balances each category from within the subject itself by oversampling, for every subject in the
        compacted_subj_dict. Each sample is its own numpy array; samples from the same category are not grouped
        under the same numpy array.

        Args:
            compacted_subj_dict (dict): a dictionary where the key is a string with the subject's name, and the value
            is its corresponding Subject object whose categories have been compacted.

        Returns:
            category_balanced_dict (dict): the balanced dictionary

        """

        category_balanced_dict = {}
        for subj_name, subject in subject_feature_dataset.items():

            category_to_idx_ls = utils.find_indices_of_duplicates(subject.categories)

            # find the category with highest representation
            max_cat, max_idx_ls = next(iter(category_to_idx_ls.items()))
            for cat_name, idx_ls in category_to_idx_ls.items():
                if len(idx_ls) > len(max_idx_ls):
                    max_cat, max_idx_ls = cat_name, idx_ls

            # determine the number of samples to add per category
            balanced_subj_data = []
            balanced_subj_cat = []
            for cat_name, idx_ls in category_to_idx_ls.items():
                num_to_add = len(max_idx_ls) - len(idx_ls)

                # convert list to numpy array to be compatible with _oversample_category() function
                cat_data = [subject.data[x] for x in idx_ls]
                cat_data_3D = np.stack(cat_data, axis=0)
                oversampled_cat = self._oversample_category(cat_data_3D, cat_name, num_to_add)
                for i in range(oversampled_cat.shape[0]):
                    sample = oversampled_cat[i, :, : ]
                    balanced_subj_data.append(sample)
                    balanced_subj_cat.append(cat_name)

            new_subj = copy(subject)  # copy the current subject
            new_subj.data = balanced_subj_data  # assign the above-calculated oversampled categories to it
            new_subj.categories = balanced_subj_cat  # assign the above-calculated oversampled categories to it
            category_balanced_dict[subj_name] = new_subj  # assign the Subject object to its name (unaltered)

        return category_balanced_dict


    def balance_categories(self, compacted_subj_dict):
        """
        Balances each category from within the subject itself by oversampling, for every subject in the
        compacted_subj_dict. All the samples from one category are contained in one numpy array in this construct.

        Args:
            compacted_subj_dict (dict): a dictionary where the key is a string with the subject's name, and the value
            is its corresponding Subject object whose categories have been compacted.

        Returns:
            category_balanced_dict (dict): the balanced dictionary

        """
        category_balanced_dict = {}
        for subj_name, subject in compacted_subj_dict.items():
            cat_data = subject.data
            cat_names = subject.categories

            # for every subject ensure that their categories are compacted i.e: the name of one category does not
            # appear more than once in the category list of the subject
            assert len(cat_names) == len(set(cat_names)), "Subject categories are not compact"

            print("\n\n", subj_name)
            max_nchunks = 0
            for i in range(0, len(cat_data)):  # find the max number of chunks within the subject's categories
                print(cat_names[i], "->", cat_data[i].shape)
                if cat_data[i].shape[0] > max_nchunks:
                    max_nchunks = cat_data[i].shape[0]
            print("Max number of chunks per category: ", max_nchunks)

            oversampled_categories = []
            for i in range(0, len(cat_data)):
                num_to_add = max_nchunks - cat_data[i].shape[0]  # number of chunks to add per category per subject
                oversampled_cat = self._oversample_category(cat_data[i], cat_names[i], num_to_add)
                oversampled_categories.append(oversampled_cat)

            new_subj = copy(subject)  # copy the current subject
            new_subj.data = oversampled_categories  # assign the above-calculated oversampled categories to it
            category_balanced_dict[subj_name] = new_subj  # assign the Subject object to its name (unaltered)

        return category_balanced_dict

    def _oversample_category(self, current_cat_data, cat_name, num_to_add):
        """
        Oversamples one single category of one single subject.

        Args:
            current_cat_data (ndarray): a ndarray of shape (number of chunks, samples_per_chunk, num_to_add)
            cat_name (string): name of category to be oversampled
            num_to_add (int): the number of chunks (2D ndarrays) to add to current_cat_data from within itself

        Returns:
            current_cat_data (ndarray): the oversampled current_cat_data with its own data by num_to_add 2D arrays

        """
        print("\nFor category", cat_name, "number of chunks to add is: ", num_to_add)
        print("Current category shape: ", current_cat_data.shape)

        chunks_to_add = []  # a list of ndarrays of duplicate chunks to append to the category at the end
        if current_cat_data.shape[0] >= num_to_add:  # if we need to add fewer chunks than there are instances of them
            # in the category
            for j in range(0, num_to_add):
                chunks_to_add.append(current_cat_data[j])
                print("(if) First inst:", current_cat_data[j][0])
        else:  # if we need to add more than 100% of the existing chunks
            current_num = 0
            while current_num < num_to_add:
                for j in range(0, current_cat_data.shape[0]):
                    if current_num < num_to_add:
                        chunks_to_add.append(current_cat_data[j])
                        print("(else) First inst:", current_cat_data[j][0])
                        current_num += 1

        if len(chunks_to_add) > 0:
            stacked_chunks = np.stack(chunks_to_add, axis=0)  # stack all the new chunks across a new dimension
            current_cat_data = np.concatenate((current_cat_data, stacked_chunks),
                                              axis=0)  # concatenate with the existing category across the 1st dim.

        return current_cat_data

