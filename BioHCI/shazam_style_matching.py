"""
Created: 1/24/19
© Denisa Qori McDonald 2019 All Rights Reserved
"""
from BioHCI.data.data_constructor import DataConstructor
from BioHCI.frequency_matching.spectrogram_filterer import Spectrogram_Filterer
from BioHCI.helpers.study_config import StudyConfig

import seaborn as sns
import matplotlib.pyplot as plt
from BioHCI.frequency_matching.hash_generator import HashGenerator
import pprint

def main():

	config_dir = "config_files"
	config = StudyConfig(config_dir)

	# create a template of a configuration file with all the fields initialized to None
	config.create_config_file_template()

	# the object with variable definition based on the specified configuration file. It includes data description,
	# definitions of run parameters (independent of deep definition vs not)
	parameters = config.populate_study_parameters("CTS_one_subj_firm.toml")

	data = DataConstructor(parameters)
	subject_dict = data.get_subject_dataset()

	for subj_name, subj in subject_dict.items():
		subj_data = subj.get_data()
		# subj_categories = subj.get_categories()
		for i, cat_data in enumerate(subj_data):
			if i < 37:
				fs = parameters.sampling_freq
				nfft = parameters.nfft
				nyq = fs / 2
				# freq_domain = data.fft_domain(cat_data)
				# N = freq_domain.shape[0]
				# print("N is: ", N)
				# w = blackman(N)
				# ywf = fft(freq_domain[:, 1]*w)

				# freq_domain_half = ywf[1:N//2]
				# print("freq domain half: ", freq_domain_half)
				# spectrum, freqs, t, im = plt.specgram(cat_data[:, 0], NFFT=N, Fs=1.0/N, noverlap=0, window=w)
				# choose N a power of 2 so that the 2 radix Cooley-Tukey algorithm can be used
				plt.figure(1)
				spectrum, freqs, t, im = plt.specgram(cat_data[:, 0], NFFT=nfft, Fs=fs, noverlap=0)

				# xf = np.linspace(0.0, 1.0 / (2.0 * T), N / 2)
				#
				# plt.semilogy(xf[1:N // 2], 2.0 / N * np.abs(freq_domain[1:N // 2]), '-b')
				# plt.semilogy(xf[1:N // 2], 2.0 / N * np.abs(ywf[1:N // 2]), '-r')
				# plt.legend(['FFT', 'FFT w. window'])
				# plt.grid()

				sf = Spectrogram_Filterer(spectrum)
				bands = sf.group_into_freq_bands(spectrum, num_bands=6)
				time_axis, freq_axis = sf.get_strongest_freq(bands)

				hash_gen = HashGenerator(time_axis, freq_axis, 10)
				hashes = hash_gen.hashes

				plt.figure(2)
				filtered_spec = sns.scatterplot(x=time_axis, y=freq_axis)
				# plt.show()

				# print("Strongest bins: ")
				# pprint.pprint(strongest_bins)

if __name__ == "__main__":
	main()
