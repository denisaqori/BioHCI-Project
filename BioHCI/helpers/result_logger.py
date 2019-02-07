import os
import datetime
import platform
from BioHCI.helpers import utilities as util


class Logging:
	def __init__(self, log_dir_path, parameter, data, learning_def, cross_validation):
		self._log_dir_path = util.create_dir(log_dir_path)

		self._parameter = parameter
		self._data = data
		self._learning_def = learning_def
		self._cross_validation = cross_validation

		# the file ton which log Results are to be written
		self._log_file = self.open_file()

	# write run parameters, definition definition and Results to file; files are named by network type and dataset
	# name;
	# if the file exists, append the result (don't overwrite), otherwise write to file
	def open_file(self):

		log_file_name = self._parameter.study_name + "_" + self._cross_validation.model.name + "_Log"

		log_file_path = os.path.join(self._log_dir_path, log_file_name)
		if os.path.exists(log_file_path):
			append_write = 'a'
		else:
			append_write = 'w'
		log_file = open(log_file_path, append_write)

		return log_file

	# write run information to file - model, data details as well as training and evaluation Results
	def log_to_file(self):

		self._log_file.write(str(datetime.datetime.now().strftime("%A, %d. %B %Y %I:%M%p") + "\n"))
		self._log_file.write("System information: \n" + str(platform.uname()) + "\n")

		self._log_file.write("Dataset Name: " + self._parameter.study_name + "\n")
		self._log_file.write("Model: " + self._cross_validation.model.name + "\n")
		self._log_file.write(str(self._cross_validation.model) + "\n\n")

		self._log_file.write("Number of original unprocessed attributes: " + str(self._parameter.num_attr) + "\n")
		self._log_file.write("Columns used: " + str(self._parameter.relevant_columns) + "\n\n")

		if self._parameter.neural_net:
			self._log_file.write("Was cuda used? - " + str(self._learning_def.use_cuda) + "\n")
			self._log_file.write("Number of Epochs per cross-validation pass: " +
								 str(self._learning_def.num_epochs) + "\n")
			self._log_file.write("Sequence Length: " + str(self._parameter.samples_per_chunk) + "\n")
			self._log_file.write("Learning rate: " + str(self._learning_def.learning_rate) + "\n\n")
			self._log_file.write("Batch size: " + str(self._learning_def.batch_size) + "\n\n")
			self._log_file.write("Dropout Rate: " + str(self._learning_def.dropout_rate) + "\n\n")

		# some evaluation metrics
		self._log_file.write("Training loss of last epoch (avg over cross-validation folds): {0}\n".format(str(
			self._cross_validation.get_avg_train_losses()[-1])))

		if self._parameter.neural_net:
			self._log_file.write("All fold train accuracies (all epochs): " +
								 str(self._cross_validation.get_all_epoch_train_accuracies()) + "\n\n")

		self._log_file.write("All fold train accuracies: " + str(self._cross_validation.get_all_train_accuracies()) +
							 "\n")
		self._log_file.write("Average train accuracy: " + str(self._cross_validation.get_avg_train_accuracy()) +
							 "\n\n")
		self._log_file.write(
			"All fold validation accuracies: " + str(self._cross_validation.get_all_val_accuracies()) + "\n")
		self._log_file.write("Average validation accuracy: " + str(self._cross_validation.get_avg_val_accuracy()) +
							 "\n\n")

		# adding performance information
		self._log_file.write("Performance Metrics:\n")
		self._log_file.write("Number of threads: " + str(self._parameter.num_threads) + "\n")
		self._log_file.write("Program time: " + str(self._cross_validation.get_total_cv_time()) + "\n")
		self._log_file.write("Total cross-validation time (" + str(self._parameter.num_folds) + " Fold): " +
							 str(self._cross_validation.get_total_cv_time()) + "\n")
		self._log_file.write("Train time (over last cross-validation pass): " +
							 str(self._cross_validation.get_train_time()) + "\n")
		self._log_file.write("Test time (over last cross-validation pass): " +
							 str(self._cross_validation.get_val_time()) + "\n")
		self._log_file.write("\n*******************************************************************\n\n\n")

		self._log_file.close()
