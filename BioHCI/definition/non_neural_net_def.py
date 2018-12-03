from BioHCI.definition.learning_def import LearningDefinition
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import sys

class NonNeuralNetworkDefinition(LearningDefinition):
	def __init__(self, model_name):

		super(NonNeuralNetworkDefinition, self).__init__(model_name)
		self.__svm_gamma = 0.001
		self.__svm_C = 100

	def _build_model(self, name):
		if name == "LDA":
			model = LDA()
		elif name == "SVM":
			model = SVC(gamma=self.__svm_gamma, C=self.__svm_C)
		else:
			print("Model specified in NonNeuralNetworkDefinition object is currently undefined!")
			sys.exit()

		return model