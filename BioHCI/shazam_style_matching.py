"""
Created: 1/24/19
© Denisa Qori McDonald 2019 All Rights Reserved
"""
from BioHCI.data.data_constructor import DataConstructor
from BioHCI.data.spectrogram_filterer import Spectrogram_Filterer
from BioHCI.helpers.study_config import StudyConfig

import seaborn as sns
import pprint

def main():

	config_dir = "config_files"
	config = StudyConfig(config_dir)

	# create a template of a configuration file with all the fields initialized to None
	config.create_config_file_template()

	# the object with variable definition based on the specified configuration file. It includes data description,
	# definitions of run parameters (independent of deep definition vs not)
	parameters = config.populate_study_parameters("CTS_one_subj_variable.toml")

	data = DataConstructor(parameters)
	subject_dict = data.get_subject_dataset()

	from scipy.fftpack import fft, fftfreq, fftshift
	from scipy.signal import blackman
	import matplotlib.pyplot as plt

	for subj_name, subj in subject_dict.items():
		subj_data = subj.get_data()
		# subj_categories = subj.get_categories()
		for i, cat_data in enumerate(subj_data):
			if i < 5:
				freq_domain = data.fft_domain(cat_data)
				N = freq_domain.shape[0]
				print("N is: ", N)
				w = blackman(N)
				ywf = fft(freq_domain[:, 1]*w)

				freq_domain_half = ywf[1:N//2]
				print("freq domain half: ", freq_domain_half)
				# spectrum, freqs, t, im = plt.specgram(cat_data[:, 0], NFFT=N, Fs=1.0/N, noverlap=0, window=w)
				# choose N a power of 2 so that the 2 radix Cooley-Tukey algorithm can be used
				plt.figure(1)
				spectrum, freqs, t, im = plt.specgram(cat_data[:, 0], NFFT=1024, Fs=2000, noverlap=0)

				print("\nfreqs from specgram: ", freqs)
				print("\nfreqs shape: ", freqs.shape)
				print("\nspectrum: ", spectrum)
				print("\nspectrum shape: ", spectrum.shape)

				# xf = np.linspace(0.0, 1.0 / (2.0 * T), N / 2)
				#
				# plt.semilogy(xf[1:N // 2], 2.0 / N * np.abs(freq_domain[1:N // 2]), '-b')
				# plt.semilogy(xf[1:N // 2], 2.0 / N * np.abs(ywf[1:N // 2]), '-r')
				# plt.legend(['FFT', 'FFT w. window'])
				# plt.grid()
				# plt.show()

				sf = Spectrogram_Filterer(spectrum)
				bands = sf.group_into_freq_bands(spectrum, num_bands=6)
				strongest_bins = sf.keep_strongest_bin(bands)
				# a = sf.keep_bins_above_mean(strongest_bins)
				top_freq = sf.get_top_freq_indices(strongest_bins, num_freq=3)
				time_axis, freq_axis = sf.get_freq_coordinates(top_freq)

				plt.figure(2)
				filtered_spec = sns.scatterplot(x=time_axis, y=freq_axis)
				plt.show()

				print("Strongest bins: ")
				pprint.pprint(strongest_bins)

if __name__ == "__main__":
	main()
