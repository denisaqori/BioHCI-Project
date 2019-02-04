"""
Created: 1/31/19
© Denisa Qori 2019 All Rights Reserved
"""

import numpy as np
import matplotlib.pyplot as plt
from BioHCI.frequency_matching.hash_generator import HashGenerator
from BioHCI.frequency_matching.spectrogram_filterer import Spectrogram_Filterer
from collections import Counter


class HashMatcher():
	def __init__(self, all_subj_dict, parameters):
		self.subj_dict = all_subj_dict
		# self.spectrogram_filterer = spectrogram_filterer
		# self.hashe_generator = hash_generator
		self.parameters = parameters

		hashes, categories = self.create_hash_dataset(all_subj_dict, parameters)
		self.score_all_signals(hashes, categories)

	def create_hash_dataset(self, all_subj_dict, parameters):
		# list of lists
		all_hashes_cat = []

		# data to stack (data per button is contained separately) - subjects end up mixed together in the ultimate
		# dataset
		hashes_data_list = []

		for subj_name, subj in all_subj_dict.items():
			subj_data = subj.get_data()
			subj_cat = subj.get_categories()

			for i, cat_data in enumerate(subj_data):
				fs = parameters.sampling_freq
				nfft = parameters.nfft

				plt.figure(1)
				spectrum, freqs, t, im = plt.specgram(cat_data[:, 0], NFFT=nfft, Fs=fs, noverlap=0)

				sf = Spectrogram_Filterer(spectrum)
				bands = sf.group_into_freq_bands(spectrum, num_bands=6)
				time_axis, freq_axis = sf.get_strongest_freq(bands, num_freq=10)

				hash_gen = HashGenerator(time_axis, freq_axis, F=10)
				hashes = hash_gen.hashes

				hashes_data_list.append(hashes)
				# append list of categories to of this category to the general hash category list
				all_hashes_cat.append([subj_cat[i]] * hashes.shape[0])

			# for j in range (0, hashes.shape[0]):
			# 	all_hashes_cat.append(subj_cat[i])

		all_hashes = np.stack(hashes_data_list, axis=0)
		return all_hashes, all_hashes_cat


	def score_one_signal(self, test_signal, index_dataset):
		hashes = index_dataset[0]
		index_hashes = np.reshape(hashes, (hashes.shape[0]*hashes.shape[1], hashes.shape[2]))
		categories = [cat for subcat in index_dataset[1] for cat in subcat]

		test_hashes = test_signal[0]
		test_cat = test_signal[1][0] # category is the same all over

		# unique_hashes = np.unique(hashes, axis=0)
		# num_unique = unique_hashes.shape[0]

		db_cat = []
		for i in range (1, test_hashes.shape[0]):
			# db_idx = []
			for j in range (0, index_hashes.shape[0]):
				if np.array_equal(test_hashes[i, :], index_hashes [j, :]):
					# db_idx.append(j)
					print("length of categories: ", len(categories))
					print("j: ", j)
					db_cat.append(categories[j])
		cat_count = Counter(db_cat)

		print (test_cat, " - ", cat_count)
		return test_cat, cat_count


	def score_all_signals (self, hashes, categories):

		real_cat = []
		retured_count = []
		for i in range(0, hashes.shape[0]):
			print ("i: ", i)
			test_hashes = hashes[i, :, :]
			test_cat = categories[i]

			index_dataset = np.delete(hashes, obj=i, axis=0)
			del categories[i]

			# index_dataset_hashes = hashes[1:, :, :]
			# index_dataset_categories = categories[1:]
			test_cat, cat_count = self.score_one_signal(test_signal=[test_hashes, test_cat],
							  index_dataset=[index_dataset, categories])
			real_cat.append(test_cat)
			retured_count.append(cat_count)
		print("")


