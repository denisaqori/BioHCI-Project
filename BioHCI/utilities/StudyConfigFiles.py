from configparser import ConfigParser
from BioHCI.definition.StudyParameters import StudyParameters
import os
import errno


class StudyConfigFiles:

	def __init__(self, root_dir_path, study_parameters):
		self.root_dir = self.create_root_dir(root_dir_path)
		self.study_parameters = study_parameters

	def generate_study_config_file(self):
		config = ConfigParser()
		config['DEFAULT'] = self.study_parameters.__dict__

		with open('example.ini', 'w') as configfile:
			config.write(configfile)
		return

	def create_root_dir(self, root_dir_path):

		root_dir = os.path.abspath(os.path.join(os.pardir, root_dir_path))
		if not os.path.exists(root_dir):
			try:
				os.makedirs(root_dir)
			except OSError as exc:  # Guard against race condition
				if exc.errno != errno.EEXIST:
					raise

	def parse_study_config_file(self, filename):
		return

if __name__ == "__main__":
	print ("Testing StudyConfigFile")

	parameters = StudyParameters()
	print(parameters.__dict__)
	root_dir = "Resources/ConfigFiles"

	config = StudyConfigFiles(root_dir, parameters)
	config.generate_study_config_file()

