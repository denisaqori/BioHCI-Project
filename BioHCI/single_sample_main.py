import torch
from torch.autograd import Variable
import numpy as np
from BioHCI.data.data_constructor import DataConstructor
from BioHCI.helpers.study_config import StudyConfig

import BioHCI.helpers.utilities as utils
from os.path import join


def main():
    config_dir = "config_files"
    config = StudyConfig(config_dir)

    # the object with variable definitions based on the specified configuration file. It includes data description,
    # definitions of run parameters (independent of deep definitions vs not)
    parameters = config.populate_study_parameters("CTS_UbiComp2020_1sample.toml")
    print(parameters)

    data = DataConstructor(parameters)
    test_data = data.test_subj_dataset

    all_categories = [str(i) for i in range(0, 36)]
    for i, cat in enumerate(all_categories):
        if len(cat) == 1:
            all_categories[i] = "button00" + cat
        elif len(cat) == 2:
            all_categories[i] = "button0" + cat

    model_subdir = join(parameters.study_name, "trained_models")
    saved_model_dir = utils.create_dir(join(utils.get_root_path("saved_objects"), model_subdir))

    model_name = "LSTM-batch-128-CTS_UbiComp2020_DescType.RawData_SeqLen.ExtendEdge_lstm_stat_2000e-fold-2-10.pt"
    model_path = join(saved_model_dir, model_name)

    predicted_val = sample_val(test_data, model_path)

    print(f"Predicted Category is {predicted_val}.")


def sample_val(subj_dataset, model_path):
    data = get_all_subj_data(subj_dataset)

    # convert numpy ndarray to PyTorch tensor + cuda
    np_data = np.asarray(data, dtype=np.float32)
    data = torch.from_numpy(np_data)
    input = Variable(data.cuda())

    # load and set model to evaluation mode, ignoring layers such as dropout and batch normalization
    model_to_eval = torch.load(model_path)
    model_to_eval.eval()

    output = model_to_eval(input)
    cat = category_from_output(output)
    del input

    return cat


def get_all_subj_data(subj_dict):
    # data to stack - subjects end up mixed together in the ultimate dataset
    all_data = []

    for subj_name, subj in subj_dict.items():
        for i, data in enumerate(subj.data):
            all_data.append(data.astype(np.float32))

    return all_data


def category_from_output(output):
    top_n, top_i = output.data.topk(k=1)  # Tensor out of Variable with .data
    category_i = top_i[0].item()
    return category_i


if __name__ == "__main__":
    main()
