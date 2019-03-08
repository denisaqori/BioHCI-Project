from BioHCI.model.cross_validator import CrossValidator
import BioHCI.helpers.utilities as utils
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

class ScipyCrossValidator(CrossValidator):

	# the number for samples per step needs to not be magic (but should be able to be different from samples_per_step)
	def __init__(self, subject_dict, data_splitter, dataset_processor, feature_constructor, model, parameter,
				 learning_def, all_categories):

		super(ScipyCrossValidator, self).__init__(subject_dict, data_splitter, dataset_processor, feature_constructor,
												  model, parameter, learning_def, all_categories)

		assert (parameter.neural_net is False), "In StudyParameters, neural_net is set to True and you are " \
														"trying to instantiate a ScipyCrossValidator object!"

		self.model = learning_def.get_model()
		# if we want to input in the algorithm measurements without specifically creating features, we can use the
		# un-chunked version of dataset and labels through the unify_time_windows function
		# Otherwise, we should build features over the time window defined by one 'chunk' or samples_per_step
		# In either case, the input arguments are tensors, as returned from data_processor, while the output values
		# are numpy arrays as expected form scikit-learn
		self.__construct_features = parameter.construct_features

	def _get_data_and_labels(self, python_dataset):
		data, labels = self._data_processor.get_shuffled_dataset_and_labels(python_dataset)
		return [data, labels]


	# TODO: add train errors and loss
	def train(self, train_dataset):

		if self.__construct_features is False:
			train_data, train_labels = utils.unify_time_windows(train_dataset[0], train_dataset[1], 30)
		else:
			train_data, train_labels = utils.define_standard_features(train_dataset[0], train_dataset[1], 30)

		# build the model using training data
		print ("Training....")
		self.model.fit(train_data, train_labels)
		self.model.score(train_data, train_labels)

	# implementing the val function declared as abstract in parent class
	def val(self, val_dataset):
		if self.__construct_features is False:
			val_data, val_labels = utils.unify_time_windows(val_dataset[0], val_dataset[1], 30)
		else:
			val_data, val_labels = utils.define_standard_features(val_dataset[0], val_dataset[1], 30)

		# evaluate the model
		print ("Evaluating...")
		correct = 0
		predicted_labels = self.model.predict (val_data)

		for i in range (len(predicted_labels)):
			if (predicted_labels[i] == val_labels[i]):
				correct = correct + 1
		accuracy = correct/len(predicted_labels)
		print ("Test Accuracy for current fold: ", accuracy)

		# metrics - come back
		accuracy_sklearn =  accuracy_score(val_labels, predicted_labels)
		print ("Test Accuracy for fold as calculated by sklearn: ", accuracy_sklearn)
		assert accuracy == accuracy_sklearn, "Accuracy not calculated properly somewhere! Different Results from " \
											 "different measures"

		self._all_train_accuracies.append(accuracy)
		self._confusion_matrix = confusion_matrix(val_labels, predicted_labels)

