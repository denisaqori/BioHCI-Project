from BioHCI.definitions.learning_def import LearningDefinition


class NeuralNetworkDefinition(LearningDefinition):
    def __init__(self, input_size: int, output_size: int, use_cuda: bool) -> None:
        # hyper-parameters
        self.__num_hidden = 100  # number of nodes per hidden layer
        self.__num_epochs = 10  # number of epochs over which to train
        # self.__samples_per_seq = 25  # the number of measurements to be included in one sequence
        self.__learning_rate = 0.001  # If you set this too high, it might explode. If too low, it might not learn
        self.__batch_size = 128  # The number of instances in one batch
        self.__dropout_rate = 0.6 # dropout rate: if 0, no dropout - to be passed to the architectures learning
        self.__num_layers = 3  # number of layers of LSTM
        self.__batch_first = True
        self.__nn_name = "CNN_LSTM_cl"

        self.__use_cuda = use_cuda
        self.__input_size = input_size
        self.__output_size = output_size

        # initialize the architectures, pick the optimizer and the loss function
        # in each case batch_size is set to true, so that input and output are expected to have the batch number as
        # the first dimension (dim=0) instead of it being the second one (dim=1) which is the default

        self._all_train_losses = []

        super(NeuralNetworkDefinition, self).__init__(input_size)

    # getters - the only way to access the class attributes
    @property
    def num_hidden(self) -> int:
        return self.__num_hidden

    @property
    def num_epochs(self) -> int:
        return self.__num_epochs

    @property
    def batch_size(self) -> int:
        return self.__batch_size

    @property
    def batch_first(self) -> bool:
        return self.__batch_first

    @property
    def nn_name(self) -> str:
        return self.__nn_name

    @property
    def learning_rate(self) -> float:
        return self.__learning_rate

    @property
    def dropout_rate(self) -> float:
        return self.__dropout_rate

    @property
    def num_layers(self) -> int:
        return self.__num_layers

    @property
    def use_cuda(self) -> bool:
        return self.__use_cuda

    @property
    def input_size(self) -> int:
        return self.__input_size

    @property
    def output_size(self) -> int:
        return self.__output_size

    # setters - to be used by a UI; does not include arguments with which the object is created
    @num_hidden.setter
    def num_hidden(self, num_hidden: int) -> None:
        self.__num_hidden = num_hidden

    @input_size.setter
    def input_size(self, input_size: int) -> None:
        self.__input_size = input_size

    @output_size.setter
    def output_size(self, output_size: int) -> None:
        self.__output_size = output_size

    @num_epochs.setter
    def num_epochs(self, num_epochs: int) -> None:
        self.__num_epochs = num_epochs

    @batch_size.setter
    def batch_size(self, batch_size: int) -> None:
        self.__batch_size = batch_size

    @batch_first.setter
    def batch_first(self, batch_first: bool) -> None:
        self.__batch_first = batch_first

    @learning_rate.setter
    def learning_rate(self, learning_rate: float) -> None:
        self.__learning_rate = learning_rate

    @dropout_rate.setter
    def dropout_rate(self, dropout_rate: float) -> None:
        self.__dropout_rate = dropout_rate

    @num_layers.setter
    def num_layers(self, num_layers: int) -> None:
        self.__num_layers = num_layers
