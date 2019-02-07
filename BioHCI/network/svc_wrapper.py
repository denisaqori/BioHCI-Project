"""
Created: 2/5/19
© Denisa Qori McDonald 2019 All Rights Reserved
"""
from sklearn.svm import SVC


class SVM:
	def __init__(self, learning_def):
		self.__learning_def = learning_def
		self.__algorithm_name = "SVM"
		self.__gamma = 0.001
		self.__C = 100

		self.__algorithm = SVC(C=self.C, gamma=self.gamma)

	@property
	def C(self):
		return self.__C

	@C.setter
	def C(self, C):
		assert self.__algorithm_name is "SVM"
		assert isinstance(C, float)

	@property
	def gamma(self):
		return self.__gamma

	@gamma.setter
	def gamma(self, gamma):
		assert self.__algorithm_name is "SVM"
		assert isinstance(gamma, float)
		self.__gamma = gamma


