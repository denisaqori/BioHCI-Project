from abc import ABC, abstractmethod


class DataSplitter(ABC):
    def __init__(self, subject_dictionary, train_val_percent=0.8, test_percent=0.2):
        self.subject_dictionary = subject_dictionary
        assert (
                train_val_percent + test_percent == 1), "The (train: validation: test) percentage split " \
                                                        "in DataSplitter does not add to 1."

        super(DataSplitter, self).__init__()

        self.train_val_percent = train_val_percent
        self.test_percent = test_percent

        self.__train_val_dictionary = subject_dictionary
        self.__test_dictionary = {}

    def split_dataset_raw(self, subject_dictionary, split_percent):
        return {}, {}

    def split_into_folds_raw(self, subject_dictionary, num_folds, val_index):
        pass

    def flatten_dataset(self):
        return

    def to_json(self):
        return

    def from_json(self):
        return

    @property
    def train_val_dict(self):
        return self.__train_val_dictionary

    @property
    def test_dict(self):
        return self.__test_dictionary

    def split_into_folds_features(self, feature_dictionary, num_folds, val_index):
        pass

    def split_dataset_features(self, feature_dataset, test_percent):
        pass

