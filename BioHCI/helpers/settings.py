"""
Created: 11/18/18
© Andrew W.E. McDonald 2018 All Rights Reserved
© Denisa Qori McDonald 2018 All Rights Reserved

Thanks to: https://btmiller.com/2015/04/11/simple-config-setup-python-yaml.html
"""
import sys
import toml
from BioHCI.helpers.utilities import time_now

'''
To use this with an associated .toml file,
you can set variables by following the dictionary/tree structure in the .toml file,
and calling the class with each argument as the next level of the dictionary in the .toml file.

E.g., if your toml file is:
#------------
[Outer]
firstVar = "my first variable"
#------------

Then use this class via:

first_var = AureliusSettings("Outer", "firstVar")

and first_var will hold "my first variable"
'''


class Configuration:
	# __metaclass__ = AureliusSettingsMeta

	_loaded = False
	aureliusSettings = {}
	configFileUsed = ""

	'''
	This overrides the __new__ behavior, so that __call__ is returned instead of a new object.
	See: https://stackoverflow.com/questions/26793600/decorate-call-with-staticmethod
	'''

	def __new__(me, *argv):
		return me.__call__(*argv)

	@classmethod
	def load(me, tomlConfigFile="./src/aurelius/config/aurelius_config.toml"):
		if not me._loaded:
			with open(tomlConfigFile) as f:
				me.aureliusSettings = toml.load(tomlConfigFile)
				me.configFileUsed = tomlConfigFile
				me._loaded = True

	@classmethod
	def __call__(me, *argv):
		if not me._loaded:
			me.load()
		tempSettingsVar = me.aureliusSettings
		# Now we loop through the input arguments -- each one must be a descendent of the previous one
		for arg in argv:
			try:
				tempSettingsVar = tempSettingsVar[arg]
			except KeyError:
				f = open("ser/AureliusSettings_" + time_now() + ".log", 'w')
				f.write("ERROR: No key '{}' in '{}' in config file: {}".format(arg, argv, me.configFileUsed))
				f.close()
				print("AureliusSettings value error! Check ./ser dir -- timestamp: {}".format(time_now()),
					  file=sys.stderr)
				return ""
		return tempSettingsVar
