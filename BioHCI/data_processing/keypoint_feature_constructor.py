"""
Created: 5/7/19
© Denisa Qori McDonald 2019 All Rights Reserved
"""
from BioHCI.data.data_constructor import DataConstructor
from BioHCI.data_processing.dataset_processor import DatasetProcessor
from BioHCI.data_processing.feature_constructor import FeatureConstructor
from BioHCI.data_processing.keypoint_description.desc_type import DescType
from BioHCI.data_processing.within_subject_oversampler import WithinSubjectOversampler
from BioHCI.helpers.study_config import StudyConfig
from BioHCI.data_processing.keypoint_description.descriptor_computer import DescriptorComputer
from BioHCI.definition.study_parameters import StudyParameters
import BioHCI.helpers.type_aliases as types
from typing import Optional


class KeypointFeatureConstructor(FeatureConstructor):
    def __init__(self, dataset_processor: DatasetProcessor, parameters: StudyParameters,
                 descriptor_computer: DescriptorComputer) -> None:
        super().__init__(dataset_processor, parameters)
        print("MSBSD Feature Constructor being initiated...")

        self.descriptor_computer = descriptor_computer

    def _produce_specific_features(self, subject_dataset: types.subj_dataset) -> Optional[types.subj_dataset]:
        # run dataset_processor on subject_dataset to compact, chunk the dataset.
        # processed_dataset = self.dataset_processor.process_dataset(subject_dataset)
        feature_dataset = self.descriptor_computer.produce_dataset_descriptors(subject_dataset)
        return feature_dataset


if __name__ == "__main__":
    print("Running msbsd_feature_constructor module...")

    config_dir = "config_files"
    config = StudyConfig(config_dir)

    # create a template of a configuration file with all the fields initialized to None
    config.create_config_file_template()
    parameters = config.populate_study_parameters("CTS_5taps_per_button.toml")

    # generating the data from files
    data = DataConstructor(parameters)
    subject_dict = data.get_subject_dataset()

    category_balancer = WithinSubjectOversampler()
    dataset_processor = DatasetProcessor(parameters, balancer=category_balancer)

    descriptor_computer = DescriptorComputer(DescType.JUSD, parameters, normalize=True,
                                             dataset_desc_name="_test")
    feature_constructor = KeypointFeatureConstructor(dataset_processor, parameters, descriptor_computer)
    feature_dataset = feature_constructor.produce_feature_dataset(subject_dict)
    print("")
