from BioHCI.definition.study_parameters import StudyParameters
from BioHCI.helpers import utilities as util
import os
import toml


class StudyConfigFiles:

	def __init__(self, root_dir_path, study_parameters):
		self.root_dir = util.create_dir(root_dir_path)
		print("Complete path to dir: ", self.root_dir)

		self.configFileUsed = None
		self.settings_dict = None
		self.__loaded = False

	def load(self, toml_config_file):
		assert toml_config_file.endswith('.toml'), "The configuration file to loads needs to have a .toml extension."
		if not self.__loaded:
			abs_configfile = os.path.join(self.root_dir, toml_config_file)
			assert os.path.isfile(abs_configfile), "The " + abs_configfile + " configuration file does not exist."
			with open(abs_configfile, 'r') as configfile:
				self.settings_dict = toml.load(configfile)
				self.configFileUsed = abs_configfile
				self.__loaded = True

	def save(self, study_parameters, toml_config_file):
		assert toml_config_file.endswith('.toml'), "The configuration file to save to needs to have a .toml extension."
		abs_configfile = os.path.join(self.root_dir, toml_config_file)
		with open(abs_configfile, 'w') as configfile:
			attr_val = vars(study_parameters)  # using vars() instead of <obj>.__dict__ because it includes
			# variables with NoneType, unlike __dict__. toml.dump() however, optimizes them out - workaround: assign
			# "None" to those variables and convert to NoneType when reading config file.
			toml.dump(attr_val, configfile)
		return

	# TODO: populate and return a study parameters object
	def populate_study_parameters(self):
		return

	# TODO: return a toml config file with all the necessary attributes empty/set to None by default
	def create_config_file_template(self):
		return

if __name__ == "__main__":
	print("Testing StudyConfigFile")

	parameters = StudyParameters()

	config_dir = "config_files"
	config = StudyConfigFiles(config_dir, parameters)

	config.save(parameters, "example2_save.toml")
	config.load("example2_save.toml")
