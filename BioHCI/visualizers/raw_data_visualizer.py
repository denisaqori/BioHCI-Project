import seaborn as sns
from BioHCI.data.data_constructor import DataConstructor
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image
import pandas as pd
import os
import math
from BioHCI.helpers import utilities as util
from BioHCI.helpers.study_config import StudyConfig


# this class takes as input a dictionary of subjects and has options to visualize that data in various ways
# this data is supposed to not have been processed for machine definition by chunking or feature construction yet.
class RawDataVisualizer:
	"""

	"""

	def __init__(self, subject_dict, category_names, xlabel, ylabel, saveplot_dir_path, verbose=False):

		print("\nVisualizing the dataset through several plots...")
		self.__verbose = verbose
		print("Verbose is set to: ", verbose)
		# a dictionary mapping one subject by name to his/her data
		self.__subject_dict = subject_dict

		# a list of category names:
		self.__category_names = category_names
		# plot labels
		self.__xlabel = xlabel
		self.__ylabel = ylabel

		# a dictionary: subject name -> subject pandas dataframe (all data in one, with category column added)
		self.__dataframe_dict = self.__create_dataframe_dict()

		self.__allsubj_dataframe = self.__create_allsubj_dataframe()

		self.__saveplot_dir_path = saveplot_dir_path

		# path to store the subject-centric plots (each plot consists of a subject's feature values per category)
		self.__subject_view = "subject_view"
		# path to store the category-centric plots (each plot consists of all subject values for a given
		# category)
		self.__category_view = "category_view"
		# path to store the combined plots in the above directories to produce summaries
		self.__combined_plots = "combined_plots"

		# create those directories whose paths have been defined above (including base saveplot_dir_path)
		subdir_name_list = [self.__subject_view, self.__category_view, self.__combined_plots]
		self.root_dir = util.create_dir(saveplot_dir_path, subdir_name_list=subdir_name_list)
		print("Look at ", saveplot_dir_path, " for such Results!")

	# creates a dictionary keyed by subject name, pointing to a dataframe of all that subject's data
	# basically, call __compact_subj_dataframe on each subject and store that information in the new dictionary
	def __create_dataframe_dict(self):
		dataframe_dict = {}
		for subj_name, subj in self.__subject_dict.items():
			if self.__verbose:
				print("\n\nSubject: ", subj_name)
			dataframe_dict[subj_name] = self.__compact_subj_dataframe(subj)
		return dataframe_dict

	# this function puts the data from one subject in one dataframe (adding a column corresponding to category)
	def __compact_subj_dataframe(self, subject):
		'''

		Args:
			subject:

		Returns:

		'''
		subj_data = subject.get_data()
		# each column will have it's own plot named after it
		category_names = subject.get_categories()

		category_list = []

		# convert the data from each category to pandas dataframe, as expected from seaborn
		for i, cat_data in enumerate(subj_data):
			subj_cat_data = pd.DataFrame(data=cat_data, columns=self.__category_names)

			# create a list of categories to append to the dataframe (so we can plot by it)
			category_name_list = [category_names[i]] * cat_data.shape[0]
			subj_cat_data['category'] = category_name_list

			category_list.append(subj_cat_data)

		compacted_dataframe = pd.concat(category_list)
		if self.__verbose:
			print("All data compacted in DataFame, including categories: \n", compacted_dataframe)

		return compacted_dataframe

	def plot_all_subj_categories(self):
		'''
		Creates a figure of feature subplots for each subject for each category. These figures are automatically
		saved in "subject_view", a subdirectory of each study's dataset_plots, found in the Results directory of the
		project. Moreover, for each subject the plots per category are combined and stored in "combined_plots" under
		the same root directory.

		'''
		for subj_name, subject_dataframe in self.__dataframe_dict.items():
			# a list to contain the image filenames of all categories for the subject
			subj_category_img_list = []
			for category, subj_cat_data in subject_dataframe.groupby('category'):
				# plot all the data and name the plot by category name
				# the last column excluded is the category name (passed separately, together with subject name)
				data_to_plot = subj_cat_data.iloc[:, 0:-1]
				cat_fig = self.plot_subj_category(data_to_plot, category, subj_name)

				# save the plot in the subject_view category
				figure_name = subj_name + "_" + category + ".png"

				figure_path = os.path.abspath(os.path.join(self.root_dir, self.__subject_view, figure_name))
				cat_fig.savefig(figure_path)
				subj_category_img_list.append(figure_path)

				if self.__verbose:
					print("Plot saved as: ", figure_path)

			# create an image representative of the subject's data for all categories
			self.combine_images(subj_category_img_list, subj_name + "_all_categories.png", 5)

	def plot_subj_category(self, data_to_plot, category, subj_name):
		"""
		Creates a figure with subplots from features of one category from one subject. Helper function for
		plot_all_subj_categories() above.

		Args:
			data_to_plot: data to be plotted (from one subject)
			category: name of category to plot
			subj_name: name of subject

		Returns:
			fig: a figure with one subject's category data

		"""

		# default theme, scaling and color palate - uses the seaborn ones,but applied to matplotlib plots too
		# try context set to 'talk' also
		sns.set(context='notebook', style='darkgrid', palette='pastel', font='sans-serif', font_scale=1,
				color_codes=True,
				rc=None)

		nplot_cols = 2
		nplot_rows = len(list(data_to_plot))

		fig = plt.figure(figsize=(nplot_rows * 5, nplot_cols * 5))
		fig.suptitle(subj_name + " - " + category, fontsize=18, y=0.99)

		G = gridspec.GridSpec(nrows=nplot_rows, ncols=nplot_cols)

		# define the graph that has all the column data under the same axis
		ax1 = fig.add_subplot(G[:, 0])
		# p = sns.lineplot(data=data_to_plot, ax=ax1)
		p = data_to_plot.plot(ax=ax1)

		# plot subplots for each column separately
		for i, column in enumerate(list(data_to_plot)):
			ax2 = fig.add_subplot(G[i, 1])
			# g = sns.lineplot(data=data_to_plot[column], ax=ax2)
			g = data_to_plot[column].plot(ax=ax2)  # same as above, but without some seaborn constraints
			# change the colors and styles of the lines to match the ones in the graph where
			# they are all plotted together for coherence's sake
			g_line = g.get_lines()[0]
			g_line.set_color(p.get_lines()[i].get_color())
			g_line.set_linestyle(p.get_lines()[i].get_linestyle())

			# put the x label at the bottom of the last individual subplot
			if i == len(list(data_to_plot)) - 1:
				ax2.set_xlabel(self.__xlabel, fontsize=14, labelpad=15)

			plt.title(column)

		ax1.set_xlabel(self.__xlabel, fontsize=14, labelpad=10)
		ax1.set_ylabel(self.__ylabel, fontsize=14, labelpad=10)

		plt.tight_layout()

		if self.__verbose:
			plt.show()

		plt.close('all')
		return fig

	def __create_allsubj_dataframe(self):
		# create a pandas dataframe that has all the subject and categories out of the dictionary that has the
		# subject name as key and its dataframe as a value
		subject_dataframe_list = []
		for subj_name, subject_dataframe in self.__dataframe_dict.items():
			subject_dataframe['subj_name'] = subj_name
			subject_dataframe_list.append(subject_dataframe)

		allsubj_dataframe = pd.concat(subject_dataframe_list)
		return allsubj_dataframe

	def plot_each_category(self):
		"""
		Generates a graph for each category with all subjects plotted under the same axis for each feature. These
		figures are automatically saved in "category_view", a subdirectory of each study's dataset_plots, found in the
		Results directory of the project. Moreover, plots for all categories are combined and stored in
		"combined_plots" under the same root directory.
		"""

		# default theme, scaling and color palate - uses the seaborn ones,but applied to matplotlib plots too
		# try context set to 'talk' also
		sns.set(context='notebook', style='darkgrid', palette='pastel', font='sans-serif', font_scale=1,
				color_codes=True,
				rc=None)

		# split the dataframe by categories, and iterate over it
		for category, cat_data in self.__allsubj_dataframe.groupby('category'):
			# keep a list of the column/feature images to later combine per category
			feature_img_list = []
			# iterate over the column names of each category (exluding the two last ones: 'category', and 'subj_name')
			for column in cat_data.iloc[:, 0:-2]:
				# build a 'channel' dataframe that has the value of only one channel and the corresponding subject name
				channel = cat_data[[column, 'subj_name']]

				channel_dataframe = pd.DataFrame()
				for subj_name, subj_channel_data in channel.groupby('subj_name'):
					channel_dataframe[subj_name] = subj_channel_data.iloc[:, 0]

				# plot the subjects' feature, dropping any NAN points
				channel_dataframe = channel_dataframe.dropna()
				channel_dataframe.plot(figsize=(15, 10))

				plt.title(category + " - " + column, fontsize=20)
				plt.xlabel(self.__xlabel, fontsize=15, labelpad=15)
				plt.ylabel(self.__ylabel, fontsize=15, labelpad=15)

				figure_name = category + " - " + column + ".png"
				figure_path = os.path.abspath(os.path.join(self.root_dir, self.__category_view, figure_name))
				plt.savefig(figure_path)
				feature_img_list.append(figure_path)

				if self.__verbose:
					print("Plot saved as: ", figure_path)
					plt.show()

			self.combine_images(feature_img_list, "all_features_" + category + ".png", 4)

		plt.close('all')

	def combine_images(self, image_list, figure_names, img_per_fig=6):
		"""
		Combines a list of images into a smaller number of figures, where each figure is the collection of a subset
		of the images passed in the list. Such images are stored in "combined_pltos", a subdirectory of
		dataset_plots of
		each dataset's results.

		Args:
			image_list: The list of images to combine
			figure_names: The list of names to assign to the generated figures grouping the above images
			img_per_fig: The number of images per figure. Default is 6.

		Returns:

		"""
		assert isinstance(image_list, list) and isinstance(figure_names, list), "The parameters image_list and " \
																				"figure_names should both be lists"
		if type(img_per_fig) is tuple:
			assert isinstance(img_per_fig[0], int) and isinstance(img_per_fig[1], int), "Values within the " \
																						"img_per_fig tuple need to be " \
																						"" \
																						"integers"
			img_per_col = img_per_fig[0]  # nrows
			img_per_row = img_per_fig[1]  # ncol
		else:
			assert isinstance(img_per_fig, int), "The parameter img_per_fig should be an integer, a tuple of " \
												 "integers, or None."
			assert math.ceil(len(image_list) / img_per_fig) == len(
				figure_names), "The number of figure names in the passed list should match the number of expected " \
							   "figures."

			# determine the number of images per row and column
			img_per_col = math.ceil(math.sqrt(img_per_fig))  # = nrows
			img_per_row = math.ceil(img_per_fig / img_per_col)  # = ncols

		if self.__verbose:
			print("\nForming a mosaic of images from the list: ", image_list)

		list_of_img_lists = []
		for i in range(0, len(image_list), img_per_fig):
			list_of_img_lists.append(image_list[i: i + img_per_fig])

		for j, imgs in enumerate(list_of_img_lists):
			self.create_figure(imgs, figure_names[j], img_per_col, img_per_row)

	def create_figure(self, image_list, figure_name, img_per_col, img_per_row, save_dir=None):
		"""
		Creates one figure out of all images passed in the image list.

		Args:
			image_list: the list of images to combine in one
			figure_name: the name of the new figure
			img_per_col: number of images per column
			img_per_row: number of images per row
			save_dir: the directory where to save the new figure. Default is None, in which case the figure is saved
				in "combined_plots".

		Returns:

		"""
		# ensure the list is composed of images with the same width and height
		img0 = Image.open(image_list[0])
		# create the image to be returned whose size is width of img0 * number of figures per row, and height is
		# height of img0 * number of images per column
		new_img = Image.new('RGB', (img0.size[0] * img_per_row, img0.size[1] * img_per_col))

		# generate coordinates to place each image in the new blank image
		img_coordinates = self.generate_coordinate_list(img_per_row, img_per_col, img0.size[0], img0.size[1])

		# paste each individual image into the new image to be combined
		for i, im_name in enumerate(image_list):
			img = Image.open(im_name)
			assert (img.size == img0.size), "The sizes of images to be combined need to match."
			new_img.paste(img, img_coordinates[i])

		# save the combined image in the directory provided. If no directory is provided, save in combined_plots of
		# dataset
		if save_dir is None:
			figure_path = os.path.abspath(os.path.join(self.root_dir, self.__combined_plots, figure_name))
		else:
			assert os.path.exists(save_dir)
			figure_path = os.path.abspath(os.path.join(save_dir, figure_name))

		if self.__verbose:
			print("Plot-combination saved as: ", figure_path, "\n")
		new_img.save(figure_path)

	@staticmethod
	def generate_coordinate_list(nimg_per_row, nimg_per_column, img_width, img_height):
		"""
		Creates a coordinate list to place images in a grid (helper for combine_images())

		Args:
			nimg_per_row: number of images per row
			nimg_per_column: number of images per column
			img_width: image width
			img_height: image height

		Returns:
			coordinate_list: a list of coordinates according to which the position of images will be determined

		"""
		coordinate_list = []
		for l in range(nimg_per_column):
			yoffset = l * img_height
			for k in range(nimg_per_row):
				xoffset = k * img_width
				coordinate_list.append((xoffset, yoffset))
		return coordinate_list

	def get_CTS_column_view(self, study_name):
		"""
		Creates combined figure of images per column of a 3 x 12 button-pad, where the label is the identity of the
		button. These plots are saved in "combined_plots".

		Args:
			study_name: the study name from where to get the subject_view plots

		Returns:

		"""
		assert "CTS" in study_name, "This method is only valid for CTS dataset"

		plot_dir_path = util.get_root_path("Results")
		if plot_dir_path is not None:
			plot_dir_path = plot_dir_path + "/" + study_name + "/dataset plots/subject_view"
			img_list = util.get_files_in_dir(plot_dir_path)
			# pp.pprint(img_list)

			# figure_names_short = ["a", "b", "c", "d", "e"]
			# figure_names = []
			# for name in figure_names_short:
			# 	figure_names.append("p1_" + name + "_all_categories.png")

			dict = {}
			dict['col1'] = (1, 13, 25)
			dict['col2'] = (2, 14, 26)
			dict['col3'] = (3, 15, 27)
			dict['col4'] = (4, 16, 28)
			dict['col5'] = (5, 17, 29)
			dict['col6'] = (6, 18, 30)
			dict['col7'] = (7, 19, 31)
			dict['col8'] = (8, 20, 32)
			dict['col9'] = (9, 21, 33)
			dict['col10'] = (10, 22, 34)
			dict['col11'] = (11, 23, 35)
			dict['col12'] = (12, 24, 36)
			figure_names = sorted(dict.keys())
			print(figure_names)

			print(len(img_list))
			j = 0
			for col, buttons in dict.items():
				img_sub_list = []
				for i, elem in enumerate(buttons):
					pattern = '_' + str(elem) + '.png'
					for entry in img_list:
						if entry.endswith(pattern):
							img_sub_list.append(entry)
				print(figure_names[j])
				self.create_figure(img_sub_list, figure_names[j] + ".png", 1, 3)
				j = j + 1

	def get_conditions_across_datasets(self, path_list, label_list, save_dir):
		"""
		Combines images across datasets and saves them in a new directory under Results.

		Args:
			path_list: a list of paths of figures to combine
			label_list: a list of labels to name figures by
			save_dir: the path to the directory to save the images

		Returns:

		"""
		plot_dir_path = util.get_root_path("Results")
		if plot_dir_path is not None:
			path_files = {}
			for path in path_list:
				path = plot_dir_path + "/" + path
				assert os.path.exists(path), "Path " + path + " does not exist."
				file_ls = util.get_files_in_dir(path)
				path_files[path] = file_ls

			for label in label_list:
				img_sub_list = []
				pattern = "_" + str(label) + ".png"
				for paths, files in path_files.items():
					for f in files:
						fname = os.path.basename(f)
						if fname.endswith(pattern):
							img_sub_list.append(f)
				fig_name = os.path.join(save_dir, "p1_all_cond_" + str(label) + ".png")
				self.create_figure(img_sub_list, fig_name, 1, 3, save_dir)


if __name__ == "__main__":
	config_dir = "config_files"
	config = StudyConfig(config_dir)

	# create a template of a configuration file with all the fields initialized to None
	config.create_config_file_template()

	# the object with variable definition based on the specified configuration file. It includes data description,
	# definitions of run parameters (independent of deep definition vs not)
	parameters = config.populate_study_parameters("CTS_one_subj_firm.toml")

	data = DataConstructor(parameters)
	subject_dict = data.get_subject_dataset()
	# build a visualizer object for the class to plot the dataset in different forms
	# we use the subject dataset as a source (a dictionary subj_name -> subj data split in categories)
	saveplot_dir_path = "Results/" + parameters.study_name + "/dataset plots"
	raw_data_vis = RawDataVisualizer(subject_dict, parameters.column_names, parameters.plot_labels[0],
									 parameters.plot_labels[1], saveplot_dir_path, verbose=False)
	# visualizing data per subject
	# raw_data_vis.plot_all_subj_categories()
	# visualizing data per category
	# raw_data_vis.plot_each_category()

	# raw_data_vis.get_CTS_column_view(parameters.study_name)

	soft_path = "CTS_one_subj_soft/dataset plots/subject_view"
	firm_path = "CTS_one_subj_firm/dataset plots/subject_view"
	variable_path = "CTS_one_subj_variable/dataset plots/subject_view"

	path_list = [soft_path, firm_path, variable_path]
	label_list = [i for i in range(1, 37)]  # keys 1 to 36 (inclusive)
	plot_dir_path = util.get_root_path("Results")
	save_dir = util.create_dir(os.path.join(plot_dir_path, "CTS_across_dataset_plots"))
	raw_data_vis.get_conditions_across_datasets(path_list, label_list, save_dir)
