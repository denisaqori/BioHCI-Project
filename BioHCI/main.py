import argparse
from os.path import join

import torch

import BioHCI.helpers.utilities as utils
from BioHCI.architectures.cnn_lstm_class import CNN_LSTM_C
from BioHCI.architectures.cnn_lstm_regr import CNN_LSTM_R
from BioHCI.data.data_constructor import DataConstructor
from BioHCI.data.within_subject_splitter import WithinSubjectSplitter
from BioHCI.data_processing.keypoint_description.desc_type import DescType
from BioHCI.data_processing.keypoint_description.descriptor_computer import DescriptorComputer
from BioHCI.data_processing.keypoint_description.sequence_length import SeqLen
from BioHCI.data_processing.keypoint_feature_constructor import KeypointFeatureConstructor
from BioHCI.data_processing.within_subject_oversampler import WithinSubjectOversampler
from BioHCI.definitions import learning_def
from BioHCI.definitions.neural_net_def import NeuralNetworkDefinition
from BioHCI.helpers.study_config import StudyConfig
from BioHCI.knitted_components.uniform_touchpad import UniformTouchpad
from BioHCI.learning.knitting_cv import KnittingCrossValidator


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
    # parameters = config.populate_study_parameters("CTS_CHI2020_test.toml")
    # parameters = config.populate_study_parameters("CTS_CHI2020_train.toml")
    parameters = config.populate_study_parameters("CTS_5taps_per_button.toml")
    print(parameters)

    # generating the data from files
    data = DataConstructor(parameters)
    # during data construction under "Subject" - button000 is ignored (baseline data)
    subject_dict = data.get_subject_dataset()

    # define a data splitter object (to be used for setting aside a testing set, as well as train/validation split
    data_splitter = WithinSubjectSplitter(subject_dict)
    # data_splitter = AcrossSubjectSplitter(subject_dict)
    category_balancer = WithinSubjectOversampler()

    descriptor_computer = DescriptorComputer(DescType.RawData, subject_dict, parameters, seq_len=SeqLen.ExtendEdge,
                                             extra_name="_test02")
    feature_constructor = KeypointFeatureConstructor(parameters, descriptor_computer)
    # feature_constructor = StatFeatureConstructor(parameters, dataset_processor)

    # estimating number of resulting features based on the shape of the dataset, to be passed later to the feature
    # constructor
    any_subj_name, any_subj = next(iter(subject_dict.items()))
    num_attr = any_subj.data[0].shape[-1]
    input_size = feature_constructor.mult_attr * num_attr
    # input_size = 1456 # for mlp

    dataset_categories = data.get_all_dataset_categories()

    assert parameters.neural_net is True
    row_learning_def = NeuralNetworkDefinition(input_size=input_size, output_size=int(len(dataset_categories) / 3),
                                                use_cuda=args.cuda)
    button_learning_def = NeuralNetworkDefinition(input_size=input_size, output_size=3, use_cuda=args.cuda)

    if parameters.classification:
        row_neural_net = CNN_LSTM_C(nn_learning_def=row_learning_def)
        button_neural_net = CNN_LSTM_C(nn_learning_def=button_learning_def)
    else:
        row_neural_net = CNN_LSTM_R(nn_learning_def=row_learning_def)
        button_neural_net = CNN_LSTM_R(nn_learning_def=button_learning_def)

    if args.cuda:
        row_neural_net.cuda()
        button_neural_net.cuda()

    # cross-validation
    assert parameters.neural_net is True
    touchpad = UniformTouchpad(num_rows=12, num_cols=3, total_resistance=534675,
                               button_resistance=7810.0, inter_button_resistance=4590.0, inter_col_resistance=13033.0)
    cv = KnittingCrossValidator(subject_dict, data_splitter, feature_constructor, category_balancer, row_neural_net,
                                button_neural_net, parameters, row_learning_def, button_learning_def,
                                dataset_categories, touchpad, descriptor_computer.dataset_desc_name)

    # cv.perform_cross_validation()
    cv.train_only()

    model_subdir = parameters.study_name + "/trained_models"
    model_name = "CNN_LSTM_classification-batch-128-CTS_CHI2020_DescType.RawData_SeqLen.ExtendEdge_real_train_only.pt"
    # model_name = "CNN_LSTM_classification-batch-128-" \
    #              "CTS_CHI2020_DescType.RawData_SeqLen.ExtendEdge_classification-fold-5-5.pt"
    saved_model_path = utils.create_dir(join(utils.get_root_path("saved_objects"), model_subdir, model_name))

    cv.eval_only(model_path=saved_model_path)

    print("\nEnd of main program.")


if __name__ == "__main__":
    main()
