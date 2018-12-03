import math
import time
import datetime

import torch
import os
import errno


# this function calculates timing difference to measure how long running certain parts takes
def time_since(since):
	now = time.time()
	s = now - since
	m = math.floor(s / 60)
	s -= m * 60
	return '%dm %ds' % (m, s)

def time_now():
	#return datetime.now().strftime("%Y%m%d-%H%M%S")
	return datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")


def create_dir(root_dir_path, subdir_name_list=None):
	'''

	Args:
		root_dir_path:
		subdir_name_list:

	Returns:

	'''
	# parent directory
	root_dir = os.path.join(get_root_path("main"), root_dir_path)
	if not os.path.exists(root_dir):
		try:
			os.makedirs(root_dir)
		except OSError as exc:  # Guard against race condition
			if exc.errno != errno.EEXIST:
				raise

	# create subdirectories if there is subdir_name_list passed as a parameter
	if subdir_name_list is not None:
		for subdir_path in subdir_name_list:
			subdir = os.path.join(root_dir, subdir_path)
			if not os.path.exists(subdir):
				try:
					os.makedirs(subdir)
				except OSError as exc:  # Guard against race condition
					if exc.errno != errno.EEXIST:
						raise

	return root_dir

def get_root_path(val):
	if val.lower() == "main":
		path = '/home/denisa/GitHub/BioHCI Project/'
	elif val.lower() == "resources":
		path = '/home/denisa/GitHub/BioHCI Project/Resources'
	elif val.lower() == "Results":
		path = '/home/denisa/GitHub/BioHCI Project/Results'
	elif val.lower() == "src":
		path = '/home/denisa/GitHub/BioHCI Project/BioHCI'
	else:
		path = None
		print("Path set to None...")
	# path = os.path.abspath(os.path.join(os.getcwd(), os.pardir, os.pardir))
	return path