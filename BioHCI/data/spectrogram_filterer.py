"""
Created: 1/17/19
© Denisa Qori 2019 All Rights Reserved
"""
import numpy as np


class Spectrogram_Filterer:
	def __init__(self, spectrogram):
		self.spectrogram = spectrogram

	def group_into_freq_bands(self, spectrogram, num_bands=6):
		"""
		Groups a spectrum input into frequency bands based on the num_bands input.

		Args:
			spectrogram (ndarray): the spectrum representation of the signal
			num_bands (int): the number of groups of frequencies to produce

		Returns:
			band_dict (dict): a dictionary mapping a tuple of frequency values to the values of those frequencies
			from the spectrum.

		"""
		num_freq = spectrogram.shape[0]
		band_dict = {}
		start_freq = 0

		base_num_freq = int(num_freq / num_bands)
		extra_freq = num_freq % num_bands

		for i in range(0, num_bands):
			if extra_freq > 0:
				stop_freq = start_freq + base_num_freq + 1
				extra_freq -= 1
			else:
				stop_freq = start_freq + base_num_freq

			band_dict[(start_freq, stop_freq - 1)] = spectrogram[start_freq: (stop_freq - 1), :]
			start_freq = stop_freq

		return band_dict


	def keep_strongest_bin(self, populated_bands):
		"""
		Filters a spectrogram to keep the most salient bin in each frequency band. The measure of strength/salience
		is simply calculated as the mean of the amplitudes of a particular frequency bin across time.

		Args:
			populated_bands (dict): a dictionary mapping a tuple of frequency interval to amplitude values over time

		Returns:
			stongest_bins (dict): a dictionary of the strongest bin in each frequency band, mapping the frequency bin
			number to the values of that bin

		"""
		strongest_bins = {}
		for band, bins in populated_bands.items():
			max_bin = bins[0]
			bin_index = band[0]
			max_index = bin_index

			for bin in bins:
				strength = bin.mean(axis=0)
				if strength > max_bin.mean(axis=0):
					max_bin = bin
					max_index = bin_index
				bin_index += 1
			strongest_bins[max_index] = max_bin

		return strongest_bins


	# TODO: checkout: amplitude mass of freq 0 is much greater than the rest
	def keep_bins_above_mean(self, strongest_bins):
		"""
		Keeps the bins of frequencies that are above the mean of all the strongest bins.

		Args:
			strongest_bins (dict): a dictionary of the strongest bin in each frequency band, mapping the frequency
								   bin to the number to the values of that bin
		Returns:
			bins_above_mean (dict):
		"""
		sum = 0
		for band, bin in strongest_bins.items():
			sum = sum + bin.sum()

		mean = sum / len(strongest_bins)

		bins_above_mean = {}
		for band, bin in strongest_bins.items():
			if bin.sum() >= mean:
				bins_above_mean[band] = bin

		return bins_above_mean


	def get_top_freq_indices(self, strongest_bins, num_freq=3):
		top_freq = {}
		for freq, bin in strongest_bins.items():
			# get indices of num_freq frequencies with highest amplitude in the bin
			top_ind = np.argpartition(bin, -num_freq)[-num_freq:]
			top_freq[freq] = top_ind

		return top_freq


	def get_freq_coordinates(self, top_freq):
		time_axis = []
		freq_axis = []
		for freq, top_indices in top_freq.items():
			for index in top_indices:
				time_axis.append(freq)
				freq_axis.append(index)

		return time_axis, freq_axis
