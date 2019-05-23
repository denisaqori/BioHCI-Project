"""
Created: 2/19/19
© Denisa Qori McDonald 2019 All Rights Reserved
"""

from BioHCI.data_processing.feature_constructor import FeatureConstructor
from BioHCI.definition.study_parameters import StudyParameters
from BioHCI.helpers.study_config import StudyConfig
from BioHCI.data.data_constructor import DataConstructor
from BioHCI.data_processing.dataset_processor import DatasetProcessor
from BioHCI.data_processing.within_subject_oversampler import WithinSubjectOversampler
import numpy as np
import BioHCI.helpers.type_aliases as types
from copy import copy
from typing import Optional


class StatFeatureConstructor(FeatureConstructor):
    """
    Statistical Information:
    """

    def __init__(self, dataset_processor: DatasetProcessor, parameters: StudyParameters) -> None:
        super().__init__(dataset_processor, parameters)
        print("Statistical Feature Constructor being initiated.")

        assert parameters.construct_features is True
        assert parameters.feature_window > 0

        # methods to calculate particular features
        self.__stat_features = [self.min_features, self.max_features, self.mean_features, self.std_features]

    def _produce_specific_features(self, subject_dataset: types.subj_dataset) -> Optional[types.subj_dataset]:
        """
        Constructs features over an interval of a chunk for the whole dataset.

        For each unprocessed original feature, each function specified in self.__stat_features is applied to each
        part of the chunk.

        Returns:
            feature_dataset: A dictionary mapping a subject name to a Subject object. The data for each category of
                this subject object will have been processed to have some features calculated.

        """
        # run dataset_processor on subject_dataset to compact, chunk the dataset.
        processed_dataset = self.dataset_processor.process_dataset(subject_dataset)

        # axis along the dataset is to be chunked in order to create an extra axis for feature construction
        chunk_axis = 1
        feature_ready_dataset = self.dataset_processor.chunk_data(processed_dataset, self.parameters.feature_window,
                                                                  chunk_axis, self.parameters.feature_overlap)

        # The axis along which features are to be created. This axis will end up being collapsed
        feature_axis = 2

        feature_dataset = {}
        for subj_name, subj in feature_ready_dataset.items():
            cat_data = subj.data

            new_cat_data = []
            for cat in cat_data:
                # assert len(cat.shape) == 4, "The subj_dataset passed to create features on, should have 4 axis."
                assert feature_axis in range(-1, len(cat.shape))

                feature_cat_tuple = ()
                for feature_func in self.__stat_features:
                    features = feature_func(cat, feature_axis)
                    feature_cat_tuple = feature_cat_tuple + (features,)

                new_cat = np.stack(feature_cat_tuple, axis=feature_axis)

                new_cat_reshaped = np.reshape(new_cat, newshape=(new_cat.shape[0], new_cat.shape[1], -1))
                new_cat_data.append(new_cat_reshaped)

            new_subj = copy(subj)  # copy the current subject
            new_subj.data = new_cat_data  # assign the above-calculated feature categories
            feature_dataset[subj_name] = new_subj  # assign the Subject object to its name (unaltered)

        return feature_dataset

    @staticmethod
    def min_features(cat: np.ndarray, feature_axis: int) -> np.ndarray:
        min_array = np.amin(cat, axis=feature_axis, keepdims=False)
        return min_array

    @staticmethod
    def max_features(cat: np.ndarray, feature_axis: int) -> np.ndarray:
        max_array = np.amax(cat, axis=feature_axis, keepdims=False)
        return max_array

    @staticmethod
    def mean_features(cat: np.ndarray, feature_axis: int) -> np.ndarray:
        mean_array = np.mean(cat, axis=feature_axis, keepdims=False)
        return mean_array

    @staticmethod
    def std_features(cat: np.ndarray, feature_axis: int) -> np.ndarray:
        std_array = np.std(cat, axis=feature_axis, keepdims=False)
        return std_array

    def diff_log_mean(self, cat: np.ndarray, feature_axis: int) -> np.ndarray:
        mean_array = self.mean_features(cat, feature_axis)
        assert mean_array.shape[-1] == 2
        diff = np.log(mean_array[:, :, 0]) - np.log(mean_array[:, :, 1])

        diff = np.expand_dims(diff, axis=feature_axis)
        return diff


if __name__ == "__main__":
    print("Running feature_constructor module...")

    config_dir = "config_files"
    config = StudyConfig(config_dir)

    # create a template of a configuration file with all the fields initialized to None
    config.create_config_file_template()
    parameters = config.populate_study_parameters("EEG_Workload.toml")

    # generating the data from files
    data = DataConstructor(parameters)
    subject_dict = data.get_subject_dataset()

    category_balancer = WithinSubjectOversampler()
    dataset_processor = DatasetProcessor(parameters, balancer=category_balancer)

    feature_constructor = StatFeatureConstructor(dataset_processor, parameters)
    feature_dataset = feature_constructor.produce_feature_dataset(subject_dict)
    print("")
