from BioHCI.definitions.learning_def import LearningDefinition


class NonNeuralNetworkDefinition(LearningDefinition):
    def __init__(self, model_name):
        super(NonNeuralNetworkDefinition, self).__init__(model_name)
        self.__svm_gamma = 0.001
        self.__svm_C = 100
