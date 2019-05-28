from BioHCI.definition.study_parameters import StudyParameters
from BioHCI.helpers import utilities as util
import os
import toml


class StudyConfig:

	def __init__(self, root_dir_path):
		self.root_dir = util.create_dir(root_dir_path)
		print("Complete path to dir: ", self.root_dir)

		self.configFileUsed = None
		self.settings_dict = None
		self.__loaded = False

	def _load(self, toml_config_file):
		"""
		Parses a toml configuration file and saves its key-value pairs in a dictionary internal to the class

		Args:
			toml_config_file (str): the name of the configuration file to be loaded

		Returns:

		"""
		assert toml_config_file.endswith('.toml'), "The configuration file to loads needs to have a .toml extension."
		if not self.__loaded:
			abs_configfile = os.path.join(self.root_dir, toml_config_file)
			assert os.path.isfile(abs_configfile), "The " + abs_configfile + " configuration file does not exist."
			with open(abs_configfile, 'r') as configfile:
				self.settings_dict = toml.load(configfile)
				self.configFileUsed = abs_configfile
				self.__loaded = True

	def dump(self, study_parameters, toml_config_file):
		"""
		Creates a configuration file based on the attributes of the StudyParameters object passed.

		Args:
			study_parameters (Stu  File "/home/denisa/GitHub/BioHCI Project/BioHCI/helpers/study_config.py", line 30, in _loaddyParameters): a StudyParameters instance containing key-value pairs
			toml_config_file (str): the name of the configuraition file to be created based on the previous argument

		Returns:

		"""
		assert toml_config_file.endswith('.toml'), "The configuration file to save to needs to have a .toml extension."
		abs_configfile = os.path.join(self.root_dir, toml_config_file)
		with open(abs_configfile, 'w') as configfile:
			attr_val = vars(study_parameters)  # using vars() instead of <obj>.__dict__ because it includes
			# variables with NoneType, unlike __dict__. toml.dump() however, optimizes them out - workaround: assign
			# "None" to those variables and convert to NoneType when reading config file.
			toml.dump(attr_val, configfile)

	def populate_study_parameters(self, toml_config_file):
		"""
		Creates, populates, and returns a StudyParameters instance object, based on the key-value pairs from the
		internally specified configuration file.

		Returns:

		"""

		self._load(toml_config_file)

		sp = StudyParameters()
		# ensure a configuration file has been loaded and its key-value pairs have been stored in the dictionary
		assert self.__loaded is True, "There was no configuration file loaded to populate the StudyParameters object."
		# sets the attributes of the instance of the StudyParameters class according to the internal dictionary
		for attribute, value in self.settings_dict.items():
			if value == "None":
				value = None
			sp.__setattr__(attribute, value)
		print("\nThe StudyParameters object has been populated using the configuration file: ", self.configFileUsed)

		return sp

	def create_config_file_template(self):
		"""
		Creates a configuration file with every attribute of the StudyParameters class set to "None",
		without changing the attributes of the only (Singleton) StudyParameters instance in use. The purpose of this
		file is to help users write configuration files.

		Returns:

		"""
		study_params = StudyParameters()
		new_dict = {}
		for attr, val in vars(study_params).items():
			new_dict[attr] = "None"

		abs_configfile = os.path.join(self.root_dir, "config_file_template.toml")
		with open(abs_configfile, 'w') as configfile:
			toml.dump(new_dict, configfile)


if __name__ == "__main__":
	print("Testing StudyConfigFile")

	parameters = StudyParameters()

	config_dir = "config_files"
	config = StudyConfig(config_dir)

	sp = config.populate_study_parameters("EEG_Workload" + ".toml")
	config.create_config_file_template()
	config.dump(parameters, parameters.study_name + ".toml")
