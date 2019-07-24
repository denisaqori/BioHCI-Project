import torch
from torch.autograd import Variable

from BioHCI.definitions.neural_net_def import NeuralNetworkDefinition
from BioHCI.definitions.study_parameters import StudyParameters
from BioHCI.helpers import utilities as util
from tensorboardX import SummaryWriter
import os
import numpy as np
from typing import List


# This class is based on PyTorch sample code from Sean Robertson (Classifying Names with a Character-Level RNN)
# http://pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial.html 


class Trainer:
    def __init__(self, train_data_loader, model, optimizer, criterion, all_int_categories: np.ndarray,
                 neural_network_def: NeuralNetworkDefinition,
                 parameters: StudyParameters, summary_writer: SummaryWriter):
        print("\nInitializing Training...")

        self.__model = model
        self.__optimizer = optimizer
        self.__criterion = criterion
        self.__num_epochs = neural_network_def.num_epochs
        self.__samples_per_chunk = parameters.samples_per_chunk
        self.__batch_size = neural_network_def.batch_size
        self.__use_cuda = neural_network_def.use_cuda

        self.__all_int_categories = all_int_categories
        self.__parameters = parameters
        self.__writer = summary_writer

        self.__epoch_losses, self.__epoch_accuracies = self.__train(train_data_loader)

    # this method returns the category based on the architectures output - each category will be associated with a likelihood
    # topk is used to get the index of highest value
    def __category_from_output(self, output):
        top_n, top_i = output.data.topk(1)  # Tensor out of Variable with .data
        category_i = int(top_i[0][0])
        return self.__all_int_categories[category_i], category_i

    # this function represents the training of one step - one chunk of data (samples_per_step) with its corresponding
    # category
    # it returns the loss data and output layer, which is then interpreted to get the predicted category using the
    # category_from_output function above. The loss data is used to go back to the weights of the architectures and adjust
    # them
    def __train_chunks_in_batch(self, category_tensor, data_chunk_tensor):

        # clear accumulated gradients from previous example
        self.__optimizer.zero_grad()

        # if cuda is available, initialize the tensors there
        if self.__use_cuda:
            data_chunk_tensor = data_chunk_tensor.cuda(async=True)
            category_tensor = category_tensor.cuda(async=True)

        # turn tensors into Variables (which can store gradients) - the necessary input to our learning
        input = Variable(data_chunk_tensor)
        label = Variable(category_tensor)

        # the forward function of the learning is run every time step
        # or every chunk/sequence of data producing an output layer, and
        # a hidden layer; the hidden layer goes in the architectures its next run
        # together with a new input - workings internal to the architectures at this point
        output = self.__model(input)

        # compute loss
        loss = self.__criterion(output, label)
        # calculate gradient descent for the variables
        loss.backward()
        # execute a gradient descent step based on the gradients calculated during the .backward() operation
        # to update the parameters of our learning
        self.__optimizer.step()

        # delete variables once we are done with them to free up space
        del input
        del label

        # we return the output of the architectures, together with the loss information
        return output, float(loss.item())

    # this is the function that handles training in general, and prints statistics regarding loss, accuracies over
    # guesses
    # for each epoch; this function returns accuracies and losses over all epochs
    def __train(self, train_data_loader):
        # Keep track of losses for plotting
        current_loss = 0
        all_losses = []
        all_accuracies = []

        for epoch in range(1, self.__num_epochs + 1):
            # number of correct guesses
            correct = 0
            total = 0
            # goes through the whole training dataset in tensor chunks and batches computing output and loss
            for step, (data_chunk_tensor, category_tensor) in enumerate(train_data_loader):  # gives batch data
                # if step == 0:
                # 	input = Variable(data_chunk_tensor)
                # 	self.__writer.add_graph(self.__model, input.cuda(async =True), True)

                # data_chunk_tensor has shape (batch_size x samples_per_chunk x num_attr)
                # category_tensor has shape (batch_size)
                # batch_size is passed as an argument to train_data_loader
                category_tensor = category_tensor.long()
                data_chunk_tensor = data_chunk_tensor.float()

                output, loss = self.__train_chunks_in_batch(category_tensor, data_chunk_tensor)
                current_loss += loss

                # for every element of the batch
                for i in range(0, self.__batch_size):
                    total = total + 1
                    # calculating true category
                    guess, guess_i = self.__category_from_output(output)
                    category_i = int(category_tensor[i])

                    # print("Guess_i: ", guess_i)
                    # print("Category_i (true category): ", category_i)

                    if category_i == guess_i:
                        # print ("Correct Guess")
                        correct += 1

            for name, param in self.__model.named_parameters():
                self.__writer.add_histogram(name, param.clone().cpu().data.numpy(), epoch)

            accuracy = correct / total
            all_accuracies.append(accuracy)
            self.__writer.add_scalar('Train/Accuracy', accuracy, epoch)

            # Print epoch number, loss, accuracy, name and guess
            print_every = 1
            if epoch % print_every == 0:
                print("Epoch ", epoch, " - Loss: ", current_loss / epoch, " Accuracy: ", accuracy)

            # Add current loss avg to list of losses
            all_losses.append(current_loss / epoch)
            self.__writer.add_scalar('Train/Avg Loss', current_loss / epoch, epoch)
            current_loss = 0

        # save trained learning
        name = self.__parameters.study_name + "-" + self.__model.name + "-batch-" + str(self.__batch_size) + \
               "-seqSize-" + str(self.__samples_per_chunk) + ".pt"
        model_dir = util.create_dir("saved_objects")
        torch.save(self.__model, os.path.join(model_dir, name))

        return all_losses, all_accuracies

    @property
    def epoch_losses(self):
        return self.__epoch_losses

    @property
    def epoch_accuracies(self):
        return self.__epoch_accuracies
