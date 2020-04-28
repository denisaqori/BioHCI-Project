import argparse

import numpy as np
import torch

from BioHCI.data.across_subject_splitter import AcrossSubjectSplitter
from BioHCI.data.data_constructor import DataConstructor
from BioHCI.data_processing.keypoint_description.desc_type import DescType
from BioHCI.data_processing.keypoint_description.descriptor_computer import DescriptorComputer
from BioHCI.data_processing.keypoint_description.sequence_length import SeqLen
from BioHCI.data_processing.keypoint_feature_constructor import KeypointFeatureConstructor
from BioHCI.data_processing.within_subject_oversampler import WithinSubjectOversampler
from BioHCI.definitions.neural_net_def import NeuralNetworkDefinition
from BioHCI.helpers.study_config import StudyConfig
from BioHCI.learning.nn_analyser import NNAnalyser


def main():
    parser = argparse.ArgumentParser(description='BioHCI arguments')
    parser.add_argument('--disable-cuda', action='store_true', help='Disable CUDA')
    parser.add_argument('--visualization', action='store_true', help='Generate plots to visualize the raw data')
    parser.add_argument('--verbose', '-v', action='store_true', help='Display more details during the run')
    args = parser.parse_args()

    # checking whether cuda is available and enabled
    args.cuda = not args.disable_cuda and torch.cuda.is_available()
    print("Is cuda available?", torch.cuda.is_available())
    print("Is the option to use cuda set?", args.cuda)

    # for reproducible results
    seed = 0
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # printing style of numpy arrays
    np.set_printoptions(precision=3, suppress=True)

    config_dir = "config_files"
    config = StudyConfig(config_dir)

    # create a template of a configuration file with all the fields initialized to None
    config.create_config_file_template()

    # the object with variable definitions based on the specified configuration file. It includes data description,
    # definitions of run parameters (independent of deep definitions vs not)
    # parameters = config.populate_study_parameters("CTS_UbiComp2020.toml")
    parameters = config.populate_study_parameters("CTS_UbiComp2020_tmp.toml")
    # parameters = config.populate_study_parameters("CTS_CHI2020_train.toml")
    print(parameters)

    # generating the data from files
    data = DataConstructor(parameters)
    cv_subject_dict = data.cv_subj_dataset
    test_subject_dict = data.test_subj_dataset
    category_balancer = WithinSubjectOversampler()

    # define a data splitter object (to be used for setting aside a testing set, as well as train/validation split
    data_splitter = AcrossSubjectSplitter(cv_subject_dict)
    cv_descriptor_computer = DescriptorComputer(DescType.RawData, cv_subject_dict, parameters,
                                                seq_len=SeqLen.ExtendEdge, extra_name="_test")
    feature_constructor = KeypointFeatureConstructor(parameters, cv_descriptor_computer)

    # estimating number of resulting features based on the shape of the dataset, to be passed later to the feature
    # constructor
    input_size = estimate_num_features(cv_subject_dict, feature_constructor)
    dataset_categories = data.get_all_dataset_categories()

    assert parameters.neural_net is True
    button_learning_def = NeuralNetworkDefinition(input_size=input_size, output_size=len(dataset_categories),
                                                  use_cuda=args.cuda)
    # learning analyser
    assert parameters.neural_net is True
    analyser = NNAnalyser(data_splitter, feature_constructor, category_balancer, parameters,
                          button_learning_def, dataset_categories, cv_descriptor_computer.dataset_desc_name)

    analyser.perform_cross_validation(cv_subject_dict)
    analyser.evaluate_all_models(test_subject_dict)
    analyser.close_logger()

    print("\nEnd of main program.")


def estimate_num_features(subj_dict, feature_constructor):
    any_subj_name, any_subj = next(iter(subj_dict.items()))
    num_attr = any_subj.data[0].shape[-1]
    input_size = feature_constructor.mult_attr * num_attr
    return input_size


if __name__ == "__main__":
    main()
