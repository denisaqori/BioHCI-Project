import torch
from torch.autograd import Variable

# Class based on PyTorch sample code from Sean Robertson (Classifying Names with a Character-Level RNN)
# http://pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial.html 


class Evaluator:
	def __init__(self, test_data_loader, model_to_eval, categories, confusion, neural_network_def, summary_writer):
		print("\n\nInitializing Evaluation...")

		self.__model_to_eval = model_to_eval
		self.__categories = categories
		self.__batch_size = neural_network_def.batch_size
		
		self.__test_data_loader = test_data_loader
		self.__use_cuda = neural_network_def.use_cuda
		self.__writer = summary_writer

		# accuracy of evaluation
		self.__accuracy = self.evaluate(self.__test_data_loader, confusion)

	# returns output layer given a tensor of data
	def evaluate_chunks_in_batch(self, data_chunk_tensor):
		# if cuda is available, initialize the tensors there
		if self.__use_cuda:
			data_chunk_tensor = data_chunk_tensor.cuda(async=True)

		# turn tensors into Variables (which can store gradients) - the necessary input to our model
		input = Variable(data_chunk_tensor)
		output = self.__model_to_eval(input)

		# delete input after we are done with it to free up space
		del input
		return output

	def evaluate(self, test_data_loader, confusion):
		# number of correct guesses
		correct = 0
		total = 0
		# Go through the test dataset and record which are correctly guessed
		for step, (data_chunk_tensor, category_tensor) in enumerate(test_data_loader):

			# data_chunk_tensor has shape (batch_size x samples_per_step x num_attr)
			# category_tensor has shape (batch_size)
			# batch_size is passed as an argument to train_data_loader
			category_tensor = category_tensor.long()
			data_chunk_tensor = data_chunk_tensor.float()

			# getting the network guess for the category
			output = self.evaluate_chunks_in_batch(data_chunk_tensor)

			# for every element of the batch
			for i in range(0, self.__batch_size):
				total = total + 1
				# calculating true category
				guess, guess_i = self.category_from_output(output)
				category_i = int(category_tensor[i])

				print("Guess_i: ", guess_i)
				print("Category_i (true category): ", category_i)

				# adding data to the matrix
				confusion[category_i][guess_i] += 1

				if category_i == guess_i:
					print ("Correct Guess")
					correct += 1

		accuracy = correct / total
		print("The number of correct guesses in the test set is:", correct, "out of", total, "total samples")
		print("Accuracy for this evaluation is: ", accuracy)
		return accuracy

	# this method returns the predicted category based on the network output - each category will be associated with a
	# likelihood
	# topk is used to get the index of highest value
	def category_from_output(self, output):
		top_n, top_i = output.data.topk(1)  # Tensor out of Variable with .data
		category_i = int(top_i[0][0])
		return self.__categories[category_i], category_i

	def get_accuracy(self):
		return self.__accuracy

