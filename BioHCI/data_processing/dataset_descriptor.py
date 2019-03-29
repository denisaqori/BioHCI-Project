"""
Created: 3/28/19
© Denisa Qori McDonald 2019 All Rights Reserved
"""
import numpy as np
import torch
import os
from BioHCI.data_processing.interval_descriptor import IntervalDescription
from BioHCI.helpers import utilities as utils
from BioHCI.data.data_constructor import DataConstructor
from BioHCI.helpers.study_config import StudyConfig
from BioHCI.data_processing.dataset_processor import DatasetProcessor
from copy import copy
import pickle

class DatasetDescriptor:
	def __init__(self, subject_dataset, desc_type, parameters, dataset_desc_name=None):
		self.subject_dataset = subject_dataset
		self.desc_type = desc_type
		self.dataset_desc_name = dataset_desc_name
		self.parameters = parameters

		dataset_desc_path = '/home/denisa/GitHub/BioHCI Project/BioHCI/data_processing/dataset_descriptors'
		self.all_dataset_desc_dir = utils.create_dir(dataset_desc_path)

	def produce_unprocessed_dataset_descriptors(self):
		descriptor_subj_dataset = {}
		for subj_name, subj in self.subject_dataset.items():
			subj_data = subj.get_data()
			subj_keypress_desc = []
			for i, keypress in enumerate(subj_data):
				interval_desc_list = IntervalDescription(keypress, self.desc_type).descriptors
				subj_keypress_desc.append(interval_desc_list)

			subj_keypress_desc = [desc for sublist in subj_keypress_desc for desc in sublist]
			new_subj = copy(subj)
			new_subj.set_data(subj_keypress_desc)
			descriptor_subj_dataset[subj_name] = new_subj

		self.save_to_file(descriptor_subj_dataset)
		return descriptor_subj_dataset


	def save_to_file(self, obj):
		"""
		Returns the path to the numpy array containing the dataset description whose name is passed as an
		argument

		Args:
		dataset_desc_name: The name of the dataset descriptors whose path is to be returned

		Returns:
			dataset_desc_path: the absolute path to that numpy array containing dataset descriptors
		"""
		dataset_desc_path = os.path.abspath(os.path.join(self.all_dataset_desc_dir, self.parameters.study_name +
							"_desc_type_" + str(self.desc_type)))
		if self.dataset_desc_name is not None:
			dataset_desc_path = dataset_desc_path + self.dataset_desc_name + ".pkl"
		else:
			dataset_desc_path = dataset_desc_path + ".pkl"

		with open(dataset_desc_path, 'wb') as f:
			pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

		return dataset_desc_path


if __name__ == "__main__":
	print("Running feature_constructor module...")
	print("Is cuda available?", torch.cuda.is_available())

	config_dir = "config_files"
	config = StudyConfig(config_dir)

	# create a template of a configuration file with all the fields initialized to None
	config.create_config_file_template()
	parameters = config.populate_study_parameters("CTS_5taps_per_button.toml")

	# generating the data from files
	data = DataConstructor(parameters)
	subject_dict = data.get_subject_dataset()

	dataset_descriptor = DatasetDescriptor(subject_dict, 2, parameters)

	dataset_processor = DatasetProcessor(parameters)
	all_desc = dataset_descriptor.produce_unprocessed_dataset_descriptors()

	print("")