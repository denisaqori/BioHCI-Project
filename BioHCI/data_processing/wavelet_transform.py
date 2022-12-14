import torch
import argparse
import numpy as np

import BioHCI.helpers.type_aliases as types
import pywt
import matplotlib.pyplot as plt

from BioHCI.data.data_constructor import DataConstructor
from BioHCI.helpers.study_config import StudyConfig


class WaveletTransform():
    def __init__(self):
        print("Wavelet Transform...")

    def lowpass_filter(self, signal: np.ndarray, filter_start: int = 4, wavelet_name: str = 'sym4') -> np.ndarray:
        """
        Deconstructs the signal using wavelets, the reconstructs it after having removed some high frequencies,
        considered noise.
        Args:
            signal: a 1D array of time-series signal to be filtered
            filter_start: the number of coefficients for each list which are kept, the rest are zeroed out as high freq.
            wavelet_name: the type of wavelet used for multi-scale analysis

        Returns: the reconstructed signal. which is the original without some high frequencies

        """
        # wavelet decomposition
        coeff = pywt.wavedec(signal, wavelet_name, mode="smooth")

        # zeroing out the last n coefficients of every level to remove high frequencies
        new_coeff = []
        for i in range(0, len(coeff)):
            if i >= filter_start:
                new_coeff.append(np.zeros(len(coeff[i])))
            else:
                new_coeff.append(coeff[i])

        # wavelet recomposition
        reconstructed_signal = pywt.waverec(new_coeff, wavelet_name, mode='smooth')
        # self.__plot_wavelet(wavelet_name, filter_start)
        # self.__plot_signals(signal, reconstructed_signal, filter_start, wavelet_name)
        return reconstructed_signal

    def __plot_wavelet(self, wavelet_name, filter_start):
        fig = plt.figure(figsize=(8, 4))
        wavelet = pywt.Wavelet(wavelet_name)
        phi, psi, x = wavelet.wavefun(level=filter_start)

        plt.plot(x, psi)
        plt.title("{}".format(wavelet_name), fontsize=16)
        plt.show()

    def __plot_signals(self, signal: np.ndarray, reconstructed_signal: np.ndarray, filter_start: int, wavelet_name: str)\
            -> None:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(signal, label='signal')
        ax.plot(reconstructed_signal, label='reconstructed signal')
        ax.legend(loc='upper left')
        ax.set_title(f"signal de- and reconstruction")# using wavedec(): {wavelet_name}, filter start: {filter_start}")
        plt.show()

    def filter_sample(self, sample: np.ndarray) -> np.ndarray:
        """
        Filters each column of the sample through wavelet decomposition - each column is considered a signal
        Args:
            sample:

        Returns: a numpy array where each column has been filtered

        """
        cols = []
        for i in range(0, sample.shape[1]):
            signal = sample[:, i]
            recon_sig = self.lowpass_filter(signal)
            cols.append(recon_sig)
        filtered_sample = np.vstack(tuple(cols))
        filtered_sample = np.swapaxes(filtered_sample, 0, 1)
        return filtered_sample

    def filter_dataset(self, dataset: types.subj_dataset) -> types.subj_dataset:
        """
        Filters the whole subject dataset using wavelets.
        Args:
            dataset: The subject dataset to be filtered

        Returns: the filtered dataset

        """
        for subj_name, subj in dataset.items():
            filtered_data = []
            for j, sample in enumerate(subj.data):
                filtered_sample = self.filter_sample(sample)
                filtered_data.append(filtered_sample)
            dataset[subj_name].data = filtered_data
        return dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='BioHCI arguments')
    parser.add_argument('--disable-cuda', action='store_true', help='Disable CUDA')
    parser.add_argument('--visualization', action='store_true', help='Generate plots to visualize the raw data')
    parser.add_argument('--verbose', '-v', action='store_true', help='Display more details during the run')
    args = parser.parse_args()

    # checking whether cuda is available and enabled
    args.cuda = not args.disable_cuda and torch.cuda.is_available()
    print("Is cuda available?", torch.cuda.is_available())
    print("Is the option to use cuda set?", args.cuda)

    # for reproducible results
    seed = 0
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # printing style of numpy arrays
    np.set_printoptions(precision=3, suppress=True)

    # """
    config_dir = "config_files"
    config = StudyConfig(config_dir)

    # create a template of a configuration file with all the fields initialized to None
    config.create_config_file_template()

    # the object with variable definitions based on the specified configuration file. It includes data description,
    # definitions of run parameters (independent of deep definitions vs not)
    # parameters = config.populate_study_parameters("CTS_UbiComp2020.toml")
    parameters = config.populate_study_parameters("CTS_4Electrodes.toml")
    print(parameters)

    # generating the data from files
    data = DataConstructor(parameters)
    subject_dict = data.cv_subj_dataset

    tr = WaveletTransform()
    tr.filter_dataset(subject_dict)