from torch.autograd import Variable


# Class based on PyTorch sample code from Sean Robertson (Classifying Names with a Character-Level RNN)
# http://pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial.html 


class Evaluator:
    def __init__(self, val_data_loader, model_to_eval, criterion, knitted_component, confusion, neural_network_def,
                 parameters, summary_writer):
        # print("\n\nInitializing Evaluation...")

        self.__model_to_eval = model_to_eval
        self.__batch_size = neural_network_def.batch_size
        self.__criterion = criterion

        self.__val_data_loader = val_data_loader
        self.__use_cuda = neural_network_def.use_cuda
        self.__writer = summary_writer
        self.__parameters = parameters

        self.__knitted_component = knitted_component
        # accuracy of evaluation
        self.__loss, self.__accuracy = self.evaluate(self.__val_data_loader, confusion)

    # returns output layer given a tensor of data
    def evaluate_chunks_in_batch(self, data_chunk_tensor, category_tensor):
        # if cuda is available, initialize the tensors there
        if self.__use_cuda:
            # data_chunk_tensor = data_chunk_tensor.cuda(async=True)
            data_chunk_tensor = data_chunk_tensor.cuda()

        # turn tensors into Variables (which can store gradients) - the necessary input to our learning
        input = Variable(data_chunk_tensor.cuda())
        label = Variable(category_tensor.cuda())

        output = self.__model_to_eval(input)
        # compute loss
        loss = self.__criterion(output, label)

        # delete input after we are done with it to free up space
        return output, float(loss.item())

    def evaluate(self, val_data_loader, confusion):
        # number of correct guesses
        correct = 0
        total = 0
        loss = 0

        # Go through the test dataset and record which are correctly guessed
        for step, (data_chunk_tensor, category_tensor) in enumerate(val_data_loader):

            # data_chunk_tensor has shape (batch_size x samples_per_step x num_attr)
            # category_tensor has shape (batch_size)
            # batch_size is passed as an argument to train_data_loader
            if self.__parameters.classification:
                category_tensor = category_tensor.long()  # the loss function requires it
            else:
                category_tensor = category_tensor.float()
            data_chunk_tensor = data_chunk_tensor.float()

            # getting the architectures guess for the category
            output, loss = self.evaluate_chunks_in_batch(data_chunk_tensor, category_tensor)
            loss += loss

            # for every element of the batch
            for i in range(0, len(category_tensor)):
                total = total + 1
                # calculating true category
                category_i = int(category_tensor[i])

                # calculating predicted categories for the whole batch
                if self.__parameters.classification:
                    predicted_i = self.category_from_output(output[i])
                else:
                    predicted_i = self.__category_from_knitted_component(output[i])

                # adding data to the matrix
                confusion[category_i][predicted_i] += 1

                if category_i == predicted_i:
                    # print("Correct Guess")
                    correct += 1

        accuracy = correct / total
        # print("The number of correct guesses in the test set is:", correct, "out of", total, "total samples")
        return loss, accuracy

    # this method returns the predicted category based on the architectures output - each category will be associated
    # with a likelihood topk is used to get the index of highest value
    def category_from_output(self, output):
        top_n, top_i = output.data.topk(k=1)  # Tensor out of Variable with .data
        category_i = top_i[0].item()
        return category_i

    def __category_from_knitted_component(self, output):
        return self.__knitted_component.get_button_id(output)

    @property
    def accuracy(self):
        return self.__accuracy

    @property
    def loss(self):
        return self.__loss
