from abc import ABC
import BioHCI.helpers.type_aliases as types
from BioHCI.definitions.study_parameters import StudyParameters
from typing import Optional


class FeatureConstructor(ABC):
    def __init__(self, parameters: StudyParameters):
        assert (parameters.construct_features is True)
        assert (parameters.feature_window is not None), "In order for features to be created, the feature window " \
                                                        "attribute should be set to an integer greater than 0, " \
                                                        "and be of NoneType."
        self.parameters = parameters
        self.__feature_dataset = None

    # def produce_feature_dataset(self, subject_dataset: types.subj_dataset) -> types.subj_dataset:
    def produce_feature_dataset(self, subject_dataset: types.subj_dataset) -> types.subj_dataset:
        """
        Constructs features on subject data based on the initialized FeatureConstructor object.

        Args:
            subject_dataset (dict): A dictionary mapping from subject name to Subject object, which contains
                                    each subject's data.

        Returns:
            feature_dataset (dict): A dictionary mapping a subject name to a Subject object. The data for each
                                    category of this subject object will have been processed to have some features
                                    calculated. These features are constructed according to the specific
                                    FeatureConstructor.
        """
        assert subject_dataset is not None, "subject_dataset needs to be set."

        feature_dataset = self._produce_specific_features(subject_dataset)
        assert feature_dataset is not None, "Class FeatureConstructor is Abstract and should not be initiated - " \
                                            "initiate one of its children instead. The produced feature_dataset is " \
                                            "currently set to None."

        self.__feature_dataset = feature_dataset

        # assert that the number of produced features is as expected, based on self.mult_attr
        assert self.mult_attr is not None
        any_subj_name, any_subj = next(iter(subject_dataset.items()))
        num_attr = any_subj.data[0].shape[-1]
        assert self.num_features == self.mult_attr * num_attr, "The shape of the returned feature dataset is not as " \
                                                               "expected: the number of colomns needs to be an " \
                                                               "integer multiple (self.mult_attr) of the number of " \
                                                               "colomns of the data for each subject unprocessed."
        return feature_dataset

    def _produce_specific_features(self, processed_dataset: types.subj_dataset) -> Optional[types.subj_dataset]:
        return None

    @property
    def feature_dataset(self) -> Optional[types.subj_dataset]:
        return self.__feature_dataset

    @property
    def num_features(self) -> Optional[int]:
        assert self.__feature_dataset is not None, "The feature dataset needs to have been created by this point"
        any_subj_name, any_subj = next(iter(self.__feature_dataset.items()))
        num = any_subj.data[0].shape[-1]
        return num

    @property
    def mult_attr(self) -> Optional[int]:
        return None
