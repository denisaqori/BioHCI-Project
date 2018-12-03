import seaborn as sns
from BioHCI.helpers import utilities as util
from BioHCI.data.data_constructor import DataConstructor
from BioHCI.definition.study_parameters import StudyParameters
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image
import pandas as pd
import os
import math


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

	# this function creates a figure of category subplots per subject
	def plot_all_subj_categories(self):
		'''

		Returns:

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
			self.combine_images(subj_category_img_list, subj_name + "_all_categories.png")

	def plot_subj_category(self, data_to_plot, category, subj_name):

		# default theme, scaling and color palate - uses the seaborn ones,but applied to matplotlib plots too
		# try context set to 'talk' also
		sns.set(context='notebook', style='darkgrid', palette='pastel', font='sans-serif', font_scale=1,
				color_codes=True,
				rc=None)

		nplot_cols = 2
		nplot_rows = len(list(data_to_plot))

		fig = plt.figure(figsize=(nplot_rows*5, nplot_cols*5))
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

	# this function generates a graph for each category with all subjects plotted under the same axis
	def plot_each_category(self):
		# create a pandas dataframe that has all the subject and categories out of the dictionary that has the
		# subject name as key and its dataframe as a value
		subject_dataframe_list = []
		for subj_name, subject_dataframe in self.__dataframe_dict.items():
			subject_dataframe['subj_name'] = subj_name
			subject_dataframe_list.append(subject_dataframe)

		allsubj_dataframe = pd.concat(subject_dataframe_list)

		# default theme, scaling and color palate - uses the seaborn ones,but applied to matplotlib plots too
		# try context set to 'talk' also
		sns.set(context='notebook', style='darkgrid', palette='pastel', font='sans-serif', font_scale=1,
				color_codes=True,
				rc=None)

		# split the dataframe by categories, and iterate over it
		for category, cat_data in allsubj_dataframe.groupby('category'):
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

			self.combine_images(feature_img_list, "all_features_" + category + ".png")

		plt.close('all')

	def combine_images(self, image_list, figure_name):
		if len(image_list) <= 25:
			if self.__verbose:
				print("\nForming a mosaic of images from the list: ", image_list)

			# determine the number of images per row and column
			img_per_col = math.ceil(math.sqrt(len(image_list)))  # = nrows
			img_per_row = math.ceil(len(image_list)/img_per_col)  # = ncols

			# ensure the list is composed of images with the same width and height
			img0 = Image.open(image_list[0])
			# create the image to be returned whose size is width of img0 * number of figures per row, and height is
			# height of img0 * number of images per column
			new_img = Image.new('RGB', (img0.size[0]*img_per_row, img0.size[1]*img_per_col))

			# generate coordinates to place each image in the new blank image
			img_coordinates = self.generate_cordinate_list(img_per_row, img_per_col, img0.size[0], img0.size[1])

			# paste each individual image into the new image to be combined
			for i, im_name in enumerate(image_list):
				img = Image.open(im_name)
				assert(img.size == img0.size), "The sizes of images to be combined need to match."
				new_img.paste(img, img_coordinates[i])

			# save the combined image
			figure_path = os.path.abspath(os.path.join(self.root_dir, self.__combined_plots, figure_name))
			if self.__verbose:
				print("Plot-combination saved as: ", figure_path, "\n")
			new_img.save(figure_path)

		else:
			print("There are too many images to meaningfully concatenate. Try a subset of them...")

		return

	# this function returns a coordinate list to place images in a grid (helper for combine_images)
	@staticmethod
	def generate_cordinate_list(nimg_per_row, nimg_per_column, img_width, img_height):
		coordinate_list = []
		for l in range(nimg_per_column):
			yoffset = l*img_height
			for k in range(nimg_per_row):
				xoffset = k*img_width
				coordinate_list.append((xoffset, yoffset))
		return coordinate_list


#TODO: ensure the real study data is received
if __name__ == "__main__":

	parameters = StudyParameters()

	data = DataConstructor(parameters)
	subject_dict = data.get_subject_dataset()
	# build a visualizer object for the class to plot the dataset in different forms
	# we use the subject dataset as a source (a dictionary subj_name -> subj data split in categories)
	saveplot_dir_path = "Results/" + "Preliminary CTS" + "/dataset plots"
	raw_data_vis = RawDataVisualizer(subject_dict, ['alpha', 'beta', 'delta', 'theta'], "Time",
									 "Power", saveplot_dir_path, verbose=False)
	# visualizing data per subject
	raw_data_vis.plot_all_subj_categories()
	# visualizing data per category
	raw_data_vis.plot_each_category()
