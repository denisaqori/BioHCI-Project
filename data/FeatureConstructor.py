

# TODO: implement FeatureConstructor Class
class FeatureConstructor:
	def __int__(self, parameters):

		print("Feature construction not implemented yet.... Should be explicitlt called after done with data "
			  "splitting, slicing, balancing.")

		self.parameters = parameters
		self.feature_window = parameters.get_feature_window()

	def build_features(self, subj_dataset):
		return self.feature_window
