from abc import ABC
import BioHCI.helpers.type_aliases as types
from BioHCI.data_processing.dataset_processor import DatasetProcessor
from BioHCI.definition.study_parameters import StudyParameters
from typing import Optional


class FeatureConstructor(ABC):
    def __init__(self, dataset_processor: DatasetProcessor, parameters: StudyParameters):
        assert (parameters.construct_features is True)
        assert (parameters.feature_window is not None), "In order for features to be created, the feature window " \
                                                        "attribute should be set to an integer greater than 0, " \
                                                        "and be of NoneType."
        self.parameters = parameters
        self.feature_window = parameters.feature_window
        self.dataset_processor = dataset_processor

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

        # run dataset_processor on subject_dataset to compact, chunk the dataset.
        processed_dataset = self.dataset_processor.process_dataset(subject_dataset)

        feature_dataset = self._produce_specific_features(processed_dataset)
        assert feature_dataset is not None, "Class FeatureConstructor is Abstract and should not be initiated - " \
                                            "initiate one of its children instead. The produced feature_dataset is " \
                                            "currently set to None."

        return feature_dataset

    def _produce_specific_features(self, processed_dataset: types.subj_dataset) -> Optional[types.subj_dataset]:
        return None
