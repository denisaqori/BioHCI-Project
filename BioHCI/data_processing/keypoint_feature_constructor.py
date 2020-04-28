"""
Created: 5/7/19
© Denisa Qori McDonald 2019 All Rights Reserved
"""
from typing import Optional

import BioHCI.helpers.type_aliases as types
from BioHCI.data.data_constructor import DataConstructor
from BioHCI.data_processing.feature_constructor import FeatureConstructor
from BioHCI.data_processing.keypoint_description.desc_type import DescType
from BioHCI.data_processing.keypoint_description.descriptor_computer import DescriptorComputer
from BioHCI.data_processing.stat_dataset_processor import StatDatasetProcessor
from BioHCI.data_processing.within_subject_oversampler import WithinSubjectOversampler
from BioHCI.definitions.study_parameters import StudyParameters
from BioHCI.helpers.study_config import StudyConfig


class KeypointFeatureConstructor(FeatureConstructor):
    def __init__(self, parameters: StudyParameters, descriptor_computer: DescriptorComputer) -> None:

        print("Keypoint Feature Constructor being initiated...")
        self.descriptor_computer = descriptor_computer

        super().__init__(parameters)

        if self.descriptor_computer.desc_type == DescType.MSD:
            self.__mult_attr = 8
        elif self.descriptor_computer.desc_type == DescType.MSBSD:
            self.__mult_attr = 16
        elif self.descriptor_computer.desc_type == DescType.RawData:
            self.__mult_attr = 1

        print("")

    @property
    def mult_attr(self) -> int:
        return self.__mult_attr

    def _produce_specific_features(self, subject_dataset: types.subj_dataset) -> Optional[types.subj_dataset]:
        # feature_dataset = self.descriptor_computer.dataset_descriptors
        feature_dataset = self.descriptor_computer.produce_dataset_descriptors(subject_dataset)
        return feature_dataset


if __name__ == "__main__":
    print("Running msbsd_feature_constructor module...")

    config_dir = "config_files"
    config = StudyConfig(config_dir)

    # create a template of a configuration file with all the fields initialized to None
    config.create_config_file_template()
    parameters = config.populate_study_parameters("CTS_Keyboard_simple.toml")

    # generating the data from files
    data = DataConstructor(parameters)
    subject_dict = data.get_subject_dataset()

    category_balancer = WithinSubjectOversampler()
    dataset_processor = StatDatasetProcessor(parameters, balancer=category_balancer)

    descriptor_computer = DescriptorComputer(DescType.MSD, subject_dict, parameters, normalize=True, extra_name="")
    feature_constructor = KeypointFeatureConstructor(parameters, descriptor_computer)

    feature_dataset = feature_constructor.produce_feature_dataset(subject_dict)
    num_features = feature_constructor.num_features
    print("")
