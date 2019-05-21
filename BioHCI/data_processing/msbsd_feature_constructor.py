"""
Created: 5/7/19
© Denisa Qori McDonald 2019 All Rights Reserved
"""
from BioHCI.data.data_constructor import DataConstructor
from BioHCI.data_processing.dataset_processor import DatasetProcessor
from BioHCI.data_processing.feature_constructor import FeatureConstructor
from BioHCI.data_processing.within_subject_oversampler import WithinSubjectOversampler
from BioHCI.helpers.study_config import StudyConfig
from BioHCI.data_processing.keypoint_description.descriptor_computer import DescriptorComputer
from BioHCI.definition.study_parameters import StudyParameters
import BioHCI.helpers.type_aliases as types

class MsbsdFeatureConstructor(FeatureConstructor):
    def __init__(self, dataset_processor: DatasetProcessor, parameters: StudyParameters,
                 descriptor_computer: DescriptorComputer, feature_axis: int) -> None:
        super().__init__(dataset_processor, parameters, feature_axis)
        print("MSBSD Feature Constructor being initiated...")

        self.descriptor_computer = descriptor_computer
        # self.features = [self.msbsd_desc]

    def _further_process(self, processed_dataset):
        # chunk_axis = 1
        # feature_ready_dataset = self.dataset_processor.chunk_data(processed_dataset, self.parameters.feature_window,
        #                                                           chunk_axis, self.parameters.feature_overlap)
        build_feature_axis = 1
        # return feature_ready_dataset, build_feature_axis
        return processed_dataset, build_feature_axis

    # def msbsd_desc(self, cat, feature_axis):
    #
    #     assert cat.ndim == 3, "The shape of the category on which to compute the descriptors needs to be 3."
    #     cat_keypress_desc = []
    #     for i in range(0, cat.shape[0]):
    #         interval = cat[i, :, :]
    #         interval_desc_list = IntervalDescription(interval, self.desc_type).descriptors
    #         cat_keypress_desc.append(interval_desc_list)
    #     cat_keypress_desc = [desc for sublist in cat_keypress_desc for desc in sublist]
    #
    #     print("")

    def specific_feature_dataset(self, subject_dataset: types.subj_dataset):



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

    descriptor_computer = DescriptorComputer(subject_dict, 1, parameters, normalize=True,
                                             dataset_desc_name="_test")
    feature_constructor = MsbsdFeatureConstructor(dataset_processor, parameters, descriptor_computer, feature_axis=2)
    feature_dataset = feature_constructor.produce_feature_dataset(subject_dict)
    print("")
