"""
Created: 2/19/19
© Denisa Qori McDonald 2019 All Rights Reserved
"""
from BioHCI.data_processing.feature_constructor import FeatureConstructor


class BoTWFeatureConstructor(FeatureConstructor):
	"""
	Bag of Temporal Words:
	"""
	def __init__(self, parameters, feature_axis):
		super(FeatureConstructor, self).__init__(parameters, feature_axis)
		print("Bag of Temporal Words being initiated...")


