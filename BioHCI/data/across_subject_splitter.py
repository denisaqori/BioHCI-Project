import math

from BioHCI.data.data_splitter import DataSplitter


class AcrossSubjectSplitter(DataSplitter):

    def __init__(self, subject_dictionary, train_val_percent=0.8, test_percent=0.2):
        super(AcrossSubjectSplitter, self).__init__(subject_dictionary, train_val_percent, test_percent)

    def split_dataset_raw(self, subject_dict, split_percent):
        """ Splits the dictionary passed as an argument into two sets, one to be used for training and validation,
        the other for testing.

        Data from one subject will belong to either one or the other; it will not be split between both. Useful for
        situations when subjects' states are to be predicted, rather than relatively permanent qualities. For
        individual quality prediction/classification use WithinSubjectSplitter, which splits each subject's data to be
        part of train and evaluation sets.

        Args:
            subject_dict (dict): a dictionary mapping subject name to the corresponding Subject object
            split_percent (float): the percentage of data from each category from the data of the Subject object to
            be included in the returned train_val_dict, with the rest of the data returned in the test_dict

        Returns:
            train_dict (dict): data to be used for training. A dictionary mapping subject name to its Subject object.
                All subject names from the input dictionary (keys), have been appended the string '_train'.
            eval_dict (dict): Data to be used for evaluation. A dictionary mapping subject name to its Subject object
                All subject names from the input dictionary (keys), have been appended the string '_eval'.

        """
        # calculate the number of subjects that are going to be part of the first sub_dictionary
        num_split_subj = math.floor(len(subject_dict) * split_percent)

        # ensures a reproducible order (according to keys)
        sorted_d = sorted(subject_dict.items())

        dict1 = dict(sorted_d[:num_split_subj])
        train_val_dict = {}
        for subj_name, subject in dict1.items():
            train_val_dict[subj_name + '_train_val'] = subject
            print (f"Subject {subj_name} in training dataset.")

        dict2 = dict(sorted_d[num_split_subj:])
        test_dict = {}
        for subj_name, subject in dict2.items():
            test_dict[subj_name + '_test'] = subject
            print (f"Subject {subj_name} in validation/testing dataset.")

        return train_val_dict, test_dict

    def split_into_folds_raw(self, subject_dictionary, num_folds, val_index):
        """

        Args:
            subject_dictionary:
            num_folds (int):
            val_index:

        Returns:

        """
        total_subj = len(subject_dictionary)
        assert total_subj >= num_folds, "The number of folds to split the dataset into cannot be greater than the " \
                                        "number of subjects for the AcrossSubjectSplitter"
        assert isinstance(num_folds, int)
        assert isinstance(val_index, int)
        assert val_index < num_folds, "Not enough folds to index with val_index"

        base = int(math.floor(total_subj / num_folds))
        extra = int(total_subj % num_folds)

        # assign the number of subjects for every fold to split the dictionary into
        split_num_ls = []
        for i in range(0, num_folds):
            if extra > 0:
                split_num_ls.append(base + 1)
                extra -= 1
            else:
                split_num_ls.append(base)

        print(split_num_ls)

        # sort the dictionary for reproducibility and then assign the number of subjects in the dict_list that is
        # specified in the corresponding index of split_num_ls
        dict_key_list = []
        sorted_dict_keys = sorted(subject_dictionary)
        i = 0
        for quantity in split_num_ls:
            dict_key_list.append(sorted_dict_keys[i:i + quantity])
            i += quantity

        # build dictionaries based on the calculated keys above and append all to a list
        dict_list = []
        for i, key_list in enumerate(dict_key_list):
            new_dict = {}
            for key in key_list:
                if i == val_index:
                    new_dict[key.replace("_train", "")] = subject_dictionary[key]
                else:
                    new_dict[key.replace("_val", "")] = subject_dictionary[key]
            dict_list.append(new_dict)

        # the validation dictionary will be the one specified by the val_index parameter
        # passing it to val_dict and pop it from the list
        val_dict = dict_list.pop(val_index)

        # merge the remaining dictionaries in the list and assign them to the train dictionary to be returned
        train_dict = {}
        for train_sub_dict in dict_list:
            train_dict.update(train_sub_dict)

        return train_dict, val_dict

    def split_dataset_features(self, feature_dataset, test_percent):
        return self.split_dataset_raw(feature_dataset, test_percent)

    def split_into_folds_features(self, feature_dictionary, num_folds, val_index):
        return self.split_into_folds_raw(feature_dictionary, num_folds, val_index)
