import torch
from tensorboardX import SummaryWriter
from torch.autograd import Variable
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader

from BioHCI.architectures.abstract_neural_net import AbstractNeuralNetwork
from BioHCI.definitions.neural_net_def import NeuralNetworkDefinition
from BioHCI.definitions.study_parameters import StudyParameters


# This class is based on PyTorch sample code from Sean Robertson (Classifying Names with a Character-Level RNN)
# http://pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial.html


class Trainer:
    def __init__(self, train_data_loader: DataLoader, neural_net: AbstractNeuralNetwork, optimizer: Optimizer,
                 criterion, neural_network_def: NeuralNetworkDefinition, parameters:
            StudyParameters, summary_writer: SummaryWriter, model_path: str) -> None:
        # print("\nInitializing Training...")

        self._neural_net = neural_net
        self._optimizer = optimizer
        self._criterion = criterion
        self._num_epochs = neural_network_def.num_epochs
        self._samples_per_chunk = parameters.samples_per_chunk
        self._batch_size = neural_network_def.batch_size
        self._use_cuda = neural_network_def.use_cuda
        self._model_path = model_path

        self._parameters = parameters
        self._writer = summary_writer

        self._loss, self._accuracy = self._train(train_data_loader)

    @property
    def model_path(self) -> str:
        return self._model_path

    # this method returns the category based on the architectures output - each category will be associated with a
    # likelihood
    # topk is used to get the index of highest value
    @staticmethod
    def __category_from_output(output) -> int:
        top_n, top_i = output.data.topk(k=1)  # Tensor out of Variable with .data
        predicted_i = top_i[0].item()
        return predicted_i

    # this function represents the training of one step - one chunk of data (samples_per_step) with its corresponding
    # category
    # it returns the loss data and output layer, which is then interpreted to get the predicted category using the
    # category_from_output function above. The loss data is used to go back to the weights of the architectures and
    # adjust
    # them
    def _train_chunks_in_batch(self, category_tensor, data_chunk_tensor, neural_net):

        # clear accumulated gradients from previous example
        # self.__optimizer.zero_grad()

        # if cuda is available, initialize the tensors there
        if self._use_cuda:
            # data_chunk_tensor = data_chunk_tensor.cuda(async=True)
            # category_tensor = category_tensor.cuda(async=True)
            data_chunk_tensor = data_chunk_tensor.cuda()
            category_tensor = category_tensor.cuda()

        # turn tensors into Variables (which can store gradients) - the necessary input to our learning
        input = Variable(data_chunk_tensor)
        label = Variable(category_tensor)

        # the forward function of the learning is run every time step
        # or every chunk/sequence of data producing an output layer, and
        # a hidden layer; the hidden layer goes in the architectures its next run
        # together with a new input - workings internal to the architectures at this point

        # output = self.__neural_net(input)
        output = neural_net(input)

        # compute loss
        loss = self._criterion(output, label)
        # calculate gradient descent for the variables
        loss.backward()
        # execute a gradient descent step based on the gradients calculated during the .backward() operation
        # to update the parameters of our learning
        self._optimizer.step()
        self._optimizer.zero_grad()

        # delete variables once we are done with them to free up space
        del input
        del label

        # we return the output of the architectures, together with the loss information
        return output, float(loss.item())

    def _train(self, train_data_loader):
        # Keep track of losses for plotting
        # number of correct guesses
        correct = 0
        total = 0
        loss = 0

        # goes through the whole training dataset in tensor chunks and batches computing output and loss
        for step, (data_chunk_tensor, category_tensor) in enumerate(train_data_loader):  # gives batch data
            if step == 0:
                input = Variable(data_chunk_tensor)
                self._writer.add_graph(self._neural_net, input.cuda(), True)

            # data_chunk_tensor has shape (batch_size x samples_per_chunk x num_attr)
            # category_tensor has shape (batch_size)
            # batch_size is passed as an argument to train_data_loader
            if self._parameters.classification:
                category_tensor = category_tensor.long()  # the loss function requires it
            else:
                category_tensor = category_tensor.float()

            data_chunk_tensor = data_chunk_tensor.float()

            output, loss = self._train_chunks_in_batch(category_tensor, data_chunk_tensor, self._neural_net)
            loss += loss

            # for every element of the batch
            for i in range(0, len(category_tensor)):
                total = total + 1

                # calculating predicted categories for the whole batch
                assert self._parameters.classification
                predicted_i = self.__category_from_output(output[i])

                category_i = int(category_tensor[i])
                if category_i == predicted_i:
                    correct += 1

        # for name, param in self.__neural_net.named_parameters():
        #     self.__writer.add_histogram(name, param.clone().cpu().data.numpy(), epoch)

        accuracy = correct / total
        # self.__all_accuracies.append(accuracy)

        return loss, accuracy

    @property
    def loss(self):
        return self._loss

    @property
    def accuracy(self):
        return self._accuracy
