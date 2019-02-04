"""
Created: 1/24/19
© Denisa Qori McDonald 2019 All Rights Reserved
"""
from BioHCI.data.data_constructor import DataConstructor
from BioHCI.frequency_matching.spectrogram_filterer import Spectrogram_Filterer
from BioHCI.helpers.study_config import StudyConfig

import matplotlib.pyplot as plt
from BioHCI.frequency_matching.hash_generator import HashGenerator
from BioHCI.frequency_matching.hash_matcher import HashMatcher
from copy import copy
import numpy as np

def main():

	config_dir = "config_files"
	config = StudyConfig(config_dir)

	# create a template of a configuration file with all the fields initialized to None
	config.create_config_file_template()

	# the object with variable definition based on the specified configuration file. It includes data description,
	# definitions of run parameters (independent of deep definition vs not)
	parameters_firm = config.populate_study_parameters("CTS_one_subj_firm.toml")
	data_firm = DataConstructor(parameters_firm)
	subject_dict_firm = data_firm.get_subject_dataset()
	new_subject_dict_firm = relabel_whole_dataset(subject_dict_firm, "Firm")

	parameters_soft = config.populate_study_parameters("CTS_one_subj_soft.toml")
	data_soft = DataConstructor(parameters_soft)
	subject_dict_soft = data_soft.get_subject_dataset()
	new_subject_dict_soft = relabel_whole_dataset(subject_dict_soft, "Soft")

	parameters_variable = config.populate_study_parameters("CTS_one_subj_variable.toml")
	data_variable = DataConstructor(parameters_variable)
	subject_dict_variable = data_variable.get_subject_dataset()
	new_subject_dict_variable = relabel_whole_dataset(subject_dict_variable, "Variable")

	assert parameters_firm.nfft == parameters_soft.nfft and parameters_firm.nfft == parameters_variable.nfft
	assert parameters_firm.sampling_freq == parameters_soft.sampling_freq and parameters_firm.sampling_freq == \
		   parameters_variable.sampling_freq

	parameters = parameters_firm
	all_subj_dict = {**new_subject_dict_firm, **new_subject_dict_soft, **new_subject_dict_variable}

	hash_matcher = HashMatcher(all_subj_dict, parameters)
	# hash_datast = hash_matcher.create_hash_dataset(all_subj_dict, parameters)

'''	
	for subj_name, subj in all_subj_dict.items():
		subj_data = subj.get_data()
		subj_cat = subj.get_categories()
		for i, cat_data in enumerate(subj_data):
			fs = parameters.sampling_freq
			nfft = parameters.nfft

			plt.figure(1)
			spectrum, freqs, t, im = plt.specgram(cat_data[:, 0], NFFT=nfft, Fs=fs, noverlap=0)

			sf = Spectrogram_Filterer(spectrum)
			bands = sf.group_into_freq_bands(spectrum, num_bands=6)
			time_axis, freq_axis = sf.get_strongest_freq(bands, num_freq=10)

			hash_gen = HashGenerator(time_axis, freq_axis, F=10)
			hashes = hash_gen.hashes

			plt.figure(2)
			filtered_spec = sns.scatterplot(x=time_axis, y=freq_axis)
			plt.show()
'''

# assign the same label to a whole dataset - removes any data labeled by baseline
def relabel_whole_dataset(subject_dict, new_label):
	new_subj_dict = {}
	for subj_name, subj in subject_dict.items():
		subj_data = subj.get_data()
		subj_cat = subj.get_categories()

		baseline_idx = [i for i, s in enumerate(subj_cat) if 'baseline' in s.lower()]
		for idx in baseline_idx:
			del subj_data[idx]

		new_cat = (len(subj_cat) - len(baseline_idx)) * [new_label]

		new_subj = copy(subj)  # copy the current subject
		new_subj.set_categories(new_cat)  # assign the above-assigned categories to it
		new_subj.set_data(subj_data)
		new_subj_dict[subj_name + new_label] = new_subj
	return new_subj_dict




if __name__ == "__main__":
	main()

