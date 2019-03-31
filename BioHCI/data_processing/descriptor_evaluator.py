"""
Created: 3/27/19
© Denisa Qori McDonald 2019 All Rights Reserved
"""
import numpy as np
from BioHCI.data.data_constructor import DataConstructor
from BioHCI.data_processing.descriptor_computer import DescriptorComputer
from BioHCI.helpers.study_config import StudyConfig
import pickle
from BioHCI.helpers import utilities as utils
import seaborn as sns
import matplotlib.pyplot as plt
import os


class DescriptorEvaluator:
	def __init__(self, descriptor_computer):
		if descriptor_computer.dataset_desc_path is not None and os.path.exists(
				descriptor_computer.dataset_desc_path):
			print("Loading dataset descriptors from: ", descriptor_computer.dataset_desc_path)
			self.__dataset_descriptors_dict = self.load_descriptors(descriptor_computer.dataset_desc_path)
		else:
			print("Producing dataset descriptors...")
			self.__dataset_descriptors_dict = descriptor_computer.produce_unprocessed_dataset_descriptors()
		self.descriptor_computer = descriptor_computer

		dataset_eval_path = '/home/denisa/GitHub/BioHCI Project/BioHCI/data_processing/dataset_evals'
		self.dataset_eval_dir = utils.create_dir(dataset_eval_path)

	@property
	def dataset_descriptors_dict(self):
		assert self.dataset_descriptors_dict is not None
		return self.__dataset_descriptors_dict

	def load_descriptors(self, dataset_desc_path):
		with open(dataset_desc_path, "rb") as input_file:
			dataset_desc = pickle.load(input_file)
		return dataset_desc

	def compute_heatmap(self, all_dataset_categories):

		heatmap = None
		if not os.path.exists(self.get_matrix_full_name()):
			for subj_name, subj in self.__dataset_descriptors_dict.items():
				subj_data = subj.get_data()
				subj_cat = subj.get_categories()
				subj_int_cat = utils.convert_categories(all_dataset_categories, subj_cat)

				heatmap = np.zeros((len(set(subj_int_cat)), len(set(subj_int_cat))))

				num = 0
				for i in range(0, len(subj_data) - 1):
					for j in range(0, len(subj_data) - 1):
						keypress1 = subj_data[i]
						cat1 = subj_int_cat[i]

						keypress2 = subj_data[j + 1]
						cat2 = subj_int_cat[j]
						# print("cat 1: ", cat1, "cat 2: ", cat2)
						print("Number of levenshtine dist computed (out of 16110 expected): ", num)

						lev_dist = self.levenshtein_distance(keypress1, keypress2)
						heatmap[cat1, cat2] = heatmap[cat1, cat2] + lev_dist
						num = num + 1

				self.save_obj(heatmap, ".pkl", "_matrix")

		else:
			with (open(self.get_matrix_full_name(), "rb")) as openfile:
				heatmap = pickle.load(openfile)

		if heatmap is not None:
			heatmap_fig = sns.heatmap(heatmap, cmap="YlGnBu")
			self.save_obj(heatmap_fig, ".png", "_heatmap")
		return

	def levenshtein_distance(self, keypress1, keypress2):
		lev_matrix = np.zeros((keypress1.shape[0], keypress2.shape[0]))
		for i in range(1, keypress1.shape[0]):
			for j in range(1, keypress2.shape[0]):
				kpress1_current_keypoint = keypress1[i, :]
				kpress2_current_keypoint = keypress2[j, :]
				current_dist = np.linalg.norm(kpress1_current_keypoint - kpress2_current_keypoint)

				diag = lev_matrix[i - 1, j - 1]
				left = lev_matrix[i - 1, j]
				up = lev_matrix[i, j - 1]

				lev_matrix[i, j] = min(diag, left, up) + current_dist

		# return the element of the last diagonal
		minimal_cost = lev_matrix[keypress1.shape[0] - 1, keypress2.shape[0] - 1]
		return minimal_cost

	def save_obj(self, obj, ext, extra_name=""):
		dataset_eval_path = os.path.abspath(os.path.join(self.dataset_eval_dir,
														 self.descriptor_computer.parameters.study_name +
														 "_desc_type_" + str(
														self.descriptor_computer.desc_type)) + extra_name + ext)
		if ext == ".pkl":
			with open(dataset_eval_path, 'wb') as f:
				pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
		elif ext == ".png":
			obj.figure.savefig(dataset_eval_path)
			plt.show()
			plt.close("all")
		else:
			print("Invalid extension. Object not saved!")

	def get_matrix_full_name(self):
		matrix_path = os.path.abspath(os.path.join(self.dataset_eval_dir,
														 self.descriptor_computer.parameters.study_name +
														 "_desc_type_" + str(
														self.descriptor_computer.desc_type)) + "_matrix.pkl")
		return matrix_path

if __name__ == "__main__":
	config_dir = "config_files"
	config = StudyConfig(config_dir)

	# create a template of a configuration file with all the fields initialized to None
	config.create_config_file_template()
	parameters = config.populate_study_parameters("CTS_5taps_per_button.toml")

	# generating the data from files
	data = DataConstructor(parameters)
	subject_dict = data.get_subject_dataset()

	# original BoTW descriptor compution
	# descriptor_1_computer = DescriptorComputer(subject_dict, 1, parameters)
	# descriptor_1_eval = DescriptorEvaluator(descriptor_1_computer)
	# descriptor_1_eval.compute_heatmap(data.get_all_dataset_categories())

	# altered BoTW descriptor compution
	descriptor_2_computer = DescriptorComputer(subject_dict, 2, parameters)
	descriptor_2_eval = DescriptorEvaluator(descriptor_2_computer)
	descriptor_2_eval.compute_heatmap(data.get_all_dataset_categories())

	print("")
