from configparser import ConfigParser
from BioHCI.definition.study_parameters import StudyParameters
from BioHCI.helpers import utilities as util
import os
import errno


class StudyConfigFiles:

	def __init__(self, root_dir_path, study_parameters):
		self.root_dir = self.create_root_dir(root_dir_path)

		print("Complete path to dir: ", self.root_dir)
		self.study_parameters = study_parameters

	def generate_study_config_file(self):
		config = ConfigParser()
		config['DEFAULT'] = self.study_parameters.__dict__
		config['Format'] = {}
		# config['Format'][self.study_parameters.]
		config['Data Information'] = {}
		config['Data Processing Decisions'] = {}
		config['Run Information'] = {}

		configfile_name = os.path.join(self.root_dir, self.study_parameters.get_study_name())
		with open(configfile_name + '.ini', 'w') as configfile:
			config.write(configfile)
		return

	def create_root_dir(self, root_dir_path):

		config_dir = os.path.join(util.get_root_path("main"), root_dir_path)
		if not os.path.exists(config_dir):
			try:
				os.makedirs(config_dir)
			except OSError as exc:  # Guard against race condition
				if exc.errno != errno.EEXIST:
					raise

		return config_dir

	def parse_study_config_file(self, filename):
		return


if __name__ == "__main__":
	print("Testing StudyConfigFile")

	parameters = StudyParameters()
	print(parameters.__dict__)

	config_dir = "config_files"

	config = StudyConfigFiles(config_dir, parameters)
	config.generate_study_config_file()
