"""
Created: 2/7/19
© Denisa Qori McDonald 2019 All Rights Reserved
"""

import seaborn as sns
from BioHCI.helpers import utilities as utils

# TODO: a save_name in cross validator to save a set of results consistently (relatively) - model name,
#  log file name, train loss/accuracy plots, confusion matrix, validation accuracy plots
class ResultsVisualizer:
	def __init__(self, parameters, cross_validation):
		self.__parameters = parameters
		self.__cross_validation = cross_validation
		conf_dir_path = "Results/" + self.__parameters.study_name + "/result graphs"
		self.__config_dir = utils.create_dir(conf_dir_path)

	def confusion_matrix_visualizer(self, confusion_matrix):
		ax = sns.heatmap(confusion_matrix)

		conf_matr_name = self.__parameters.study_name + "_" + self.__cross_validation.model.name + ".png"

		return ax

