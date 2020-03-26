import argparse
from os.path import join

import torch

from BioHCI.data.across_subject_splitter import AcrossSubjectSplitter
from BioHCI.data.data_constructor import DataConstructor
from BioHCI.data.within_subject_splitter import WithinSubjectSplitter
from BioHCI.data_processing.keypoint_description.desc_type import DescType
from BioHCI.data_processing.keypoint_description.descriptor_computer import DescriptorComputer
from BioHCI.data_processing.keypoint_description.sequence_length import SeqLen
from BioHCI.data_processing.keypoint_feature_constructor import KeypointFeatureConstructor
from BioHCI.data_processing.within_subject_oversampler import WithinSubjectOversampler
from BioHCI.definitions.neural_net_def import NeuralNetworkDefinition
from BioHCI.helpers.study_config import StudyConfig
from BioHCI.knitted_components.uniform_touchpad import UniformTouchpad
from BioHCI.learning.nn_cross_validator import NNCrossValidator
from BioHCI.learning.nn_msd_cv import NN_MSD_CrossValidator
import BioHCI.helpers.utilities as utils


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

    # torch.manual_seed(1)  # reproducible Results for testing purposes

    config_dir = "config_files"
    config = StudyConfig(config_dir)

    # create a template of a configuration file with all the fields initialized to None
    config.create_config_file_template()

    # TODO: save confusion matrix
    # the object with variable definitions based on the specified configuration file. It includes data description,
    # definitions of run parameters (independent of deep definitions vs not)
    # parameters = config.populate_study_parameters("CTS_CHI2020_test.toml")
    parameters = config.populate_study_parameters("CTS_CHI2020_train.toml")
    # parameters = config.populate_study_parameters("CTS_5taps_per_button.toml")
    print(parameters)

    # generating the data from files
    data = DataConstructor(parameters)
    # during data construction under "Subject" - button000 is ignored (baseline data)
    subject_dict = data.get_subject_dataset()
    category_balancer = WithinSubjectOversampler()

    # define a data splitter object (to be used for setting aside a testing set, as well as train/validation split
    # data_splitter = WithinSubjectSplitter(subject_dict)
    data_splitter = AcrossSubjectSplitter(subject_dict)

    descriptor_computer = DescriptorComputer(DescType.RawData, subject_dict, parameters, seq_len=SeqLen.ExtendEdge,
                                                # extra_name="_nocoef_nolinreg_1000e_rand")
                                                extra_name = "_test_saved_nn")
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
    # row_learning_def = NeuralNetworkDefinition(input_size=input_size, output_size=int(len(dataset_categories) / 3),
    #                                             use_cuda=args.cuda)
    button_learning_def = NeuralNetworkDefinition(input_size=input_size, output_size=36, use_cuda=args.cuda)

    # cross-validation
    assert parameters.neural_net is True
    # touchpad = UniformTouchpad(num_rows=12, num_cols=3, total_resistance=534675,
    #                            button_resistance=7810.0, inter_button_resistance=4590.0, inter_col_resistance=13033.0)
    # cv = NN_MSD_CrossValidator(subject_dict, data_splitter, feature_constructor, category_balancer, parameters,
    #                             row_learning_def, dataset_categories, touchpad, descriptor_computer.dataset_desc_name)
    cv = NNCrossValidator(subject_dict, data_splitter, feature_constructor, category_balancer, parameters,
                           button_learning_def, dataset_categories, descriptor_computer.dataset_desc_name)

    # cv.perform_cross_validation()
    # cv.train_only()

    # cm_obj_name = "CNN_LSTM_cl-batch-128-CTS_CHI2020_DescType.RawData_SeqLen.ExtendEdge_final_study1_confusion_matrix.pt"
    # cm_obj_name = "CNN_LSTM_cl-batch-128-CTS_CHI2020_DescType.RawData_SeqLen.ExtendEdge_nocoef_nolinreg_1500e_confusion_matrix.pt"
    # cv.generate_confusion_matrix_fig_from_obj_name(cm_obj_name)

    model_subdir = parameters.study_name + "/trained_models"
    # model_name = "CNN_LSTM_cl-batch-128-CTS_CHI2020_DescType.RawData_SeqLen.ExtendEdge_all_linreg_1500e-fold-10-10.pt"
    # model_name = "CNN_LSTM_cl-batch-128-CTS_CHI2020_DescType.RawData_SeqLen.ExtendEdge_final-fold-10-10.pt"
    # model_name = "CNN_LSTM_cl-batch-128-CTS_CHI2020_DescType.RawData_SeqLen.ExtendEdge_nocoef_nolinreg_1000e_rand-fold-1-10.pt"
    # model_name = "CNN_LSTM_cl-batch-128-CTS_CHI2020_DescType.RawData_SeqLen.ExtendEdge_nocoef_nolinreg_1500e-fold-10-10.pt"
    model_name = "CNN_LSTM_cl-batch-128-CTS_CHI2020_DescType.RawData_SeqLen.ExtendEdge_test_saved_nn-fold-2-10.pt"
    saved_model_path = utils.create_dir(join(utils.get_root_path("saved_objects"), model_subdir, model_name))

    cv.eval_only(model_path=saved_model_path)

    print("\nEnd of main program.")

if __name__ == "__main__":
    main()
