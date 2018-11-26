import torch
from torch.autograd import Variable
import os

# This class is based on PyTorch sample code from Sean Robertson (Classifying Names with a Character-Level RNN)
# http://pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial.html 


class Trainer:
	def __init__(self, train_data_loader, data, deep_learning_def):
		print("\nInitializing Training...")

		self._model = deep_learning_def.get_model()
		self._data = data
		self.__optimizer = deep_learning_def.get_optimizer()
		self.__criterion = deep_learning_def.get_criterion()
		self.__num_epochs = deep_learning_def.get_num_epochs()
		self.__samples_per_step = deep_learning_def.get_samples_per_step()
		self._batch_size = deep_learning_def.get_batch_size()
		self._use_cuda = deep_learning_def.is_use_cuda()

		self.__train_data_loader = train_data_loader
		self._categories = data.get_categories()

		self.__epoch_losses, self.__epoch_accuracies = self.__train(train_data_loader)

	# this method returns the category based on the network output - each category will be associated with a likelihood
	# topk is used to get the index of highest value
	def category_from_output(self, output):
		top_n, top_i = output.data.topk(1)  # Tensor out of Variable with .data
		category_i = int(top_i[0][0])
		return self._categories[category_i], category_i


	# this function represents the training of one step - one chunk of data (samples_per_step) with its corresponding category
	# it returns the loss data and output layer, which is then interpreted to get the predicted category using the 
	# category_from_output function above. The loss data is used to go back to the weights of the network and adjust them
	def __train_chunks_in_batch(self, category_tensor, data_chunk_tensor):

		# clear accumulated gradients from previous example
		self.__optimizer.zero_grad()

		# if cuda is available, initialize the tensors there
		if self._use_cuda:
			data_chunk_tensor = data_chunk_tensor.cuda(async=True)
			category_tensor = category_tensor.cuda(async=True)

		# turn tensors into Variables (which can store gradients) - the necessary input to our model
		input = Variable(data_chunk_tensor)
		label = Variable(category_tensor)

		# the forward function of the model is run every time step
		# or every chunk/sequence of data producing an output layer, and
		# a hidden layer; the hidden layer goes in the network its next run
		# together with a new input - workings internal to the network at this point
		output = self._model(input)

		# compute loss
		loss = self.__criterion(output, label)
		# calculate gradient descent for the variables
		loss.backward()
		# execute a gradient descent step based on the gradients calculated during the .backward() operation
		# to update the parameters of our model
		self.__optimizer.step()

		# delete variables once we are done with them to free up space
		del input
		del label

		# we return the output of the network, together with the loss information
		return output, float(loss.item())

	# this is the function that handles training in general, and prints statistics regarding loss, accuracies over guesses
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
			for step, (data_chunk_tensor, category_tensor) in enumerate(train_data_loader): # gives batch data

				# data_chunk_tensor has shape (batch_size x samples_per_step x num_features)
				# category_tensor has shape (batch_size)
				# batch_size is passed as an argument to train_data_loader
				category_tensor = category_tensor.long()
				data_chunk_tensor = data_chunk_tensor.float()

				output, loss = self.__train_chunks_in_batch(category_tensor, data_chunk_tensor)
				current_loss += loss

				# for every element of the batch
				for i in range(0, self._batch_size):
					total = total + 1
					# calculating true category
					guess, guess_i = self.category_from_output(output)
					category_i = int(category_tensor[i])

					# print("Guess_i: ", guess_i)
					# print("Category_i (true category): ", category_i)

					if category_i == guess_i:
						# print ("Correct Guess")
						correct += 1

			accuracy = correct / total
			all_accuracies.append(accuracy)

			# Print epoch number, loss, accuracy, name and guess
			print_every = 1
			if epoch % print_every == 0:
				print("Epoch ", epoch, " - Loss: ", current_loss/epoch, " Accuracy: ", accuracy)

			# Add current loss avg to list of losses
			all_losses.append(current_loss / epoch)
			current_loss = 0

		# save trained model
		name = self._data.get_dataset_name() + "-" + self._model.name + "-batch-" + str(self._batch_size) + \
			   "-seqSize-" \
			   + str(self.__samples_per_step) + ".pt"
		#torch.save(self.model, 'saved_models/toy-lstm-classification.pt')
		torch.save(self._model, os.path.join("saved_models", name))

		return all_losses, all_accuracies


	def get_epoch_losses(self):
		return self.__epoch_losses

	def get_epoch_accuracies(self):
		return self.__epoch_accuracies
