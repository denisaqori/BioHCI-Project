import argparse

import torch

from BioHCI.data.within_subject_splitter import WithinSubjectSplitter
from BioHCI.data.data_constructor import DataConstructor
from BioHCI.data_processing.keypoint_description.desc_type import DescType
from BioHCI.data_processing.keypoint_description.descriptor_computer import DescriptorComputer
from BioHCI.data_processing.keypoint_description.sequence_length import SequenceLength
from BioHCI.data_processing.keypoint_feature_constructor import KeypointFeatureConstructor
from BioHCI.data_processing.stat_dataset_processor import StatDatasetProcessor
from BioHCI.data_processing.stat_feature_constructor import StatFeatureConstructor
from BioHCI.data_processing.within_subject_oversampler import WithinSubjectOversampler
from BioHCI.definitions.neural_net_def import NeuralNetworkDefinition
from BioHCI.learning.nn_cross_validator import NNCrossValidator
from BioHCI.helpers.result_logger import Logging

from BioHCI.visualizers.raw_data_visualizer import RawDataVisualizer
from BioHCI.helpers.study_config import StudyConfig
from BioHCI.architectures.lstm import LSTM


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

    torch.manual_seed(1)  # reproducible Results for testing purposes

    config_dir = "config_files"
    config = StudyConfig(config_dir)

    # create a template of a configuration file with all the fields initialized to None
    config.create_config_file_template()

    # the object with variable definitions based on the specified configuration file. It includes data description,
    # definitions of run parameters (independent of deep definitions vs not)
    parameters = config.populate_study_parameters("CTS_5taps_per_button.toml")
    # parameters = config.populate_study_parameters("EEG_Workload.toml")
    print(parameters)

    # generating the data from files
    data = DataConstructor(parameters)
    subject_dict = data.get_subject_dataset()

    if args.visualization:
        # build a visualizer object for the class to plot the dataset in different forms
        # we use the subject dataset as a source (a dictionary subj_name -> subj data split in categories)
        saveplot_dir_path = "Results/" + parameters.study_name + "/dataset plots"
        raw_data_vis = RawDataVisualizer(subject_dict, parameters, saveplot_dir_path, verbose=False)
        # visualizing data per subject
        raw_data_vis.plot_all_subj_categories()
        # visualizing data per category
        raw_data_vis.plot_each_category()
        raw_data_vis.compute_spectrogram(subject_dict)

    # define a data splitter object (to be used for setting aside a testing set, as well as train/validation split
    data_splitter = WithinSubjectSplitter(subject_dict)
    category_balancer = WithinSubjectOversampler()

    descriptor_computer = DescriptorComputer(DescType.JUSD, subject_dict, parameters, normalize=True,
                                             seq_len=SequenceLength.ZeroPad, extra_name="_pipeline_test")
    #
    feature_constructor = KeypointFeatureConstructor(parameters, descriptor_computer)

    # dataset_processor = StatDatasetProcessor(parameters)
    # feature_constructor = StatFeatureConstructor(parameters, dataset_processor)

    # estimating number of resulting features based on the shape of the dataset, to be passed later to the feature
    # constructor
    any_subj_name, any_subj = next(iter(subject_dict.items()))
    num_attr = any_subj.data[0].shape[-1]
    input_size = feature_constructor.mult_attr * num_attr

    datast_categories = data.get_all_dataset_categories()

    assert parameters.neural_net is True
    learning_def = NeuralNetworkDefinition(input_size=input_size, output_size=len(datast_categories),
                                           use_cuda=args.cuda)
    neural_net = LSTM(nn_learning_def=learning_def)
    if args.cuda:
        neural_net.cuda()

    # cross-validation
    assert parameters.neural_net is True
    cv = NNCrossValidator(subject_dict, data_splitter, feature_constructor, category_balancer, neural_net, parameters,
                          learning_def, datast_categories)

    cv.perform_cross_validation()

    # results of run
    # log_dir_path = "Results/" + parameters.study_name + "/run summaries"
    # logging = Logging(log_dir_path, parameters, data, learning_def, cv)
    # logging.log_to_file()

    print("\nEnd of main program.")


if __name__ == "__main__":
    main()
