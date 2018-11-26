import torch
from torch.autograd import Variable

# Class based on PyTorch sample code from Sean Robertson (Classifying Names with a Character-Level RNN)
# http://pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial.html 


class Evaluator:
	def __init__(self, test_data_loader, model_to_eval, categories, confusion, deep_learning_def):
		print("\n\nInitializing Evaluation...")

		self._model_to_eval = model_to_eval
		self._categories = categories
		self._batch_size = deep_learning_def.get_batch_size()
		
		self._test_data_loader = test_data_loader
		self._use_cuda = deep_learning_def.is_use_cuda()

		# accuracy of evaluation
		self._accuracy = self.evaluate(self._test_data_loader, confusion)

	# returns output layer given a tensor of data
	def evaluate_chunks_in_batch(self, data_chunk_tensor):
		# if cuda is available, initialize the tensors there
		if self._use_cuda:
			data_chunk_tensor = data_chunk_tensor.cuda(async=True)

		# turn tensors into Variables (which can store gradients) - the necessary input to our model
		input = Variable(data_chunk_tensor)
		output = self._model_to_eval(input)

		# delete input after we are done with it to free up space
		del input
		return output

	def evaluate(self, test_data_loader, confusion):
		# number of correct guesses
		correct = 0
		total = 0
		# Go through the test dataset and record which are correctly guessed
		for step, (data_chunk_tensor, category_tensor) in enumerate(test_data_loader):

			# data_chunk_tensor has shape (batch_size x samples_per_step x num_features)
			# category_tensor has shape (batch_size)
			# batch_size is passed as an argument to train_data_loader
			category_tensor = category_tensor.long()
			data_chunk_tensor = data_chunk_tensor.float()

			# getting the network guess for the category
			output = self.evaluate_chunks_in_batch(data_chunk_tensor)

			# for every element of the batch
			for i in range(0, self._batch_size):
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
		return self._categories[category_i], category_i

	def get_accuracy(self):
		return self._accuracy

