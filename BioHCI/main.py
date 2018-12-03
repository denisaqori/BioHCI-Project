import argparse

import torch

from BioHCI.data.across_subject_splitter import AcrossSubjectSplitter
from BioHCI.data.within_subject_splitter import WithinSubjectSplitter
from BioHCI.data.data_constructor import DataConstructor
from BioHCI.data.dataset_processor import DatasetProcessor
from BioHCI.data.within_subject_oversampler import WithinSubjectOversampler
from BioHCI.definition.neural_net_def import NeuralNetworkDefinition
from BioHCI.definition.non_neural_net_def import NonNeuralNetworkDefinition
from BioHCI.definition.study_parameters import StudyParameters
from BioHCI.model.neural_network_cv import NeuralNetworkCV
from BioHCI.model.non_neural_network_cv import NonNeuralNetworkCV
from BioHCI.helpers.result_logger import Logging
from BioHCI.data.feature_constructor import FeatureConstructor
from BioHCI.data.data_augmenter import DataAugmenter

from BioHCI.helpers.raw_data_visualizer import RawDataVisualizer


def main():
	parser = argparse.ArgumentParser(description='BioHCI arguments')
	parser.add_argument('--disable-cuda', action='store_true', help='Disable CUDA')
	parser.add_argument('--visualization', action='store_true', help='Generate plots to visualize the raw data')
	parser.add_argument('--verbose', '-v', action='store_true', help='Display more details during the run')
	args = parser.parse_args()

	# checking whether cuda is available and enabled
	args.cuda = not args.disable_cuda and torch.cuda.is_available()
	print("Is cuda available?", torch.cuda.is_available())
	print("Is the option to use cuda set?", args.cuda)

	torch.manual_seed(1)  # reproducible Results for testing purposes

	parameters = StudyParameters()  # contains definitions of run parameters (independent of deep definition vs not
	# model)
	# the main place to define variables, including data description

	# generating the data from files
	data = DataConstructor(parameters)
	subject_dict = data.get_subject_dataset()

	if args.visualization:
		# build a visualizer object for the class to plot the dataset in different forms
		# we use the subject dataset as a source (a dictionary subj_name -> subj data split in categories)
		saveplot_dir_path = "Results/" + parameters.get_study_name() + "/dataset plots"
		raw_data_vis = RawDataVisualizer(subject_dict, ['alpha', 'beta', 'delta', 'theta'], "Time",
										 "Power", saveplot_dir_path, verbose=False)
		# visualizing data per subject
		raw_data_vis.plot_all_subj_categories()
		# visualizing data per category
		raw_data_vis.plot_each_category()

	# define a data splitter object (to be used for setting aside a testing set, as well as train/validation split
	data_splitter = AcrossSubjectSplitter(subject_dict)
	train_val_dictionary = data_splitter.get_train_val_dict()

	# define a category balancer (implementing the abstract CategoryBalancer)
	category_balancer = WithinSubjectOversampler()
	# initialize the feature constructor
	feature_constructor = FeatureConstructor(parameters)
	print("feature constructor results: ", feature_constructor.build_features(subject_dict))
	data_augmenter = DataAugmenter()

	dataset_processor = DatasetProcessor(parameters.get_samples_per_chunk(), parameters.is_interval_overlap(),
										 category_balancer, feature_constructor, data_augmenter)

	# if we want a deep definition model, define it specifically in the NeuralNetworkDefinition class
	num_categories = len(data.get_all_dataset_categories())
	if parameters.neural_net is True:
		learning_def = NeuralNetworkDefinition(model_name="CNN_LSTM", num_features=parameters.num_features,
											   output_size=num_categories, use_cuda=args.cuda)

		model = learning_def.get_model()
		print("\nNetwork Architecture: \n", model)

		if args.cuda:
			model.cuda()
	else:
		learning_def = NonNeuralNetworkDefinition(model_name="SVM")

	# cross-validation
	if parameters.neural_net is True:
		cv = NeuralNetworkCV(subject_dict, data_splitter, dataset_processor, parameters, learning_def, num_categories)
	else:
		cv = NonNeuralNetworkCV(subject_dict, data_splitter, dataset_processor, parameters, learning_def,
								num_categories)

	# results of run
	log_dir_path = "Results/" + parameters.get_study_name() + "/run summaries"
	logging = Logging(log_dir_path, parameters, data, learning_def, cv)
	logging.log_to_file()

	print("\nEnd of program for now!!!!!")


"""
_______________________________________________________________________________________________________________________
Portion below is to be removed, only here for inspiration, or maybe reusble code-blocks
_______________________________________________________________________________________________________________________
	# create a confusion matrix to track correct guesses (accumulated over all folds of the Cross-Validation below
	confusion = torch.zeros(data.get_num_categories(), data.get_num_categories())

	# running cross validation; splitting the dataset into folds, using one as testing once while training on all the
	# rest for each run, the accuracy is saved, to later be averaged with all runs
	all_test_accuracies = []
	all_train_accuracies = []
	cv_losses = []

	cv_start = time.time()
	for i in range(0, parameter.get_num_folds()):
		print(
			"\n\n"
			"*******************************************************************************************************")
		print("Run: ", i)
		train, test = data_processor.split_in_folds(k=parameter.get_num_folds(), s=i)

		print("Getting training dataset and labels...")
		# train_dataset, train_labels = data_processor.get_shuffled_dataset_and_labels(train)
		# the train_dataset is a tuple of TensorDataset type, containing a tensor with train data, and one with train
		# labels
		train_data, train_labels = data_processor.get_shuffled_dataset_and_labels(train)
		train_dataset = TensorDataset(train_data, train_labels)
		print("Using the PyTorch DataLoader to load the training data (shuffled) with: \nbatch size = ", dl.batch_size,
			  " & number of threads = ", parameter.get_num_threads())
		train_data_loader = DataLoader(train_dataset, batch_size=dl.batch_size, 
		num_workers=parameter.get_num_threads(),
									   shuffle=True, pin_memory=True)

		print("Getting testing dataset and labels...")
		# the test_dataset is a tuple containing a tensor with test data, and one with test labels (each input into the
		# evaluator later
		test_data, test_labels = data_processor.get_shuffled_dataset_and_labels(test)
		test_dataset = TensorDataset(test_data, test_labels)
		print("Using the PyTorch DataLoader to load the testing data (shuffled) with: \nbatch size = ", dl.batch_size,
			  " & number of threads = ", num_threads)
		test_data_loader = DataLoader(test_dataset, batch_size=dl.batch_size, num_workers=num_threads, shuffle=True,
									  pin_memory=True)

		# starting training with the above-defined parameters
		train_start = time.time()
		trainer = Trainer(train_data_loader, data.categories, model, data, dl.optimizer, dl.criterion, dl.num_epochs,
						  dl.samples_per_step, dl.batch_size, args.cuda)
		train_time = utils.time_since(train_start)

		# get the loss over all epochs for this cv-fold and append it to the list
		cv_losses.append(trainer.all_losses)
		print("Train Losses: ", trainer.all_losses)

		all_train_accuracies.append(trainer.all_accuracies)

		# this is the network produces by training over the other folds
		# model_to_eval = torch.load('saved_models/gender-rnn-classification.pt')
		model_name = data.get_dataset_name() + "-" + model.name + "-batch-" + str(dl.batch_size) + "-seqSize-" \
					 + str(dl.samples_per_step) + ".pt"
		model_to_eval = torch.load(os.path.join("saved_No usages founmodels", model_name))

		test_start = time.time()
		evaluator = Evaluator(test_data_loader, data.categories, model_to_eval, dl.batch_size, confusion, args.cuda)
		test_time = utils.time_since(test_start)

		fold_accuracy = evaluator.accuracy
		all_test_accuracies.append(fold_accuracy)

	cv_time = utils.time_since(cv_start)

	# plotting losses for each category over each epoch
	# this is the average graph of losses for all cross validation folds
	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.set_title('Average Loss per Epoch')

	ax.set_xlabel('Number of Epochs')
	ax.set_ylabel('Average Loss')

	avg_losses = []
	for i in range(dl.num_epochs):
		epoch_loss = 0
		for j, loss_list in enumerate(cv_losses):
			epoch_loss = epoch_loss + loss_list[i]
		avg_losses.append(epoch_loss / num_folds)

	plt.plot(avg_losses)
	plt.show()

	# save the plot
	train_epochs_path = 'Results/train loss plots/'
	# plt.savefig(os.path.join(train_epochs_path, model.name + "_Loss_" + str(num_epochs) + "Epochs"))

	print("All Losses: ", cv_losses)
	print("Avg Losses: ", avg_losses)

	# Show the Confusion Matrix for each class
	# Normalize by dividing every row by its sum
	total_sum = 0
	for i in range(len(data.categories)):
		total_sum = total_sum + confusion[i].sum()
		confusion[i] = confusion[i] / confusion[i].sum()

	print("Total Sum of evaluated chunks: ", total_sum)
	print("All fold accuracies: ", all_test_accuracies)

	# avg_accuracy = sum(all_test_accuracies) / float(len(all_test_accuracies))
	# print("\nAverage accuracy: ", avg_accuracy)

	# Set up plot
	fig = plt.figure()
	ax = fig.add_subplot(111)
	cax = ax.matshow(confusion.numpy())
	fig.colorbar(cax)

	# Set up axes
	ax.set_xticklabels([''] + data.categories, rotation=90)
	ax.set_yticklabels([''] + data.categories)

	# Force label at every tick
	ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
	ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

	# sphinx_gallery_thumbnail_number = 2
	plt.show()

	# save the results of confusion matrix
	confusion_matrix_path = 'Results/matrix eval plots/'
'''

# plt.savefig(os.path.join(confusion_matrix_path, model.name + "_Confusion_" + str(num_epochs) + "Epochs"))

"""
if __name__ == "__main__":
	main()
