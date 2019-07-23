from abc import ABC, abstractmethod


# TODO: if test_percent == 0, load test_dict from file
class DataSplitter(ABC):
    def __init__(self, subject_dictionary, train_val_percent=0.8, test_percent=0.2):
        self.subject_dictionary = subject_dictionary
        assert (
                train_val_percent + test_percent == 1), "The (train: validation: test) percentage split " \
                                                        "in DataSplitter does not add to 1."

        super(DataSplitter, self).__init__()

        self.train_val_percent = train_val_percent
        self.test_percent = test_percent

        if self.test_percent != 0:
            self.__train_val_dictionary, self.__test_dictionary = self.split_dataset(self.subject_dictionary,
                                                                                     train_val_percent)
        else:
            self.__train_val_dictionary = subject_dictionary
            self.__test_dictionary = {}

    @abstractmethod
    def split_dataset(self, subject_dictionary, split_percent):
        return {}, {}

    @abstractmethod
    def split_into_folds(self, subject_dictionary, num_folds, val_index):
        pass

    # def get_processed_datsets(self):
    # 	return self.processed_train_val_dict, self.processed_test_dict

    def flatten_dataset(self):
        return

    def to_json(self):
        return

    def from_json(self):
        return

    def get_train_val_dict(self):
        return self.__train_val_dictionary

    def get_test_dict(self):
        return self.__test_dictionary
