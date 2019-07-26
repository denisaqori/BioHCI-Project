"""
Created: 2/19/19
© Denisa Qori McDonald 2019 All Rights Reserved
"""
import math

from BioHCI.data_processing.feature_constructor import FeatureConstructor
from BioHCI.helpers.study_config import StudyConfig
from BioHCI.data.data_constructor import DataConstructor
from BioHCI.data_processing.stat_dataset_processor import StatDatasetProcessor
import BioHCI.helpers.utilities as utils
from BioHCI.data_processing.within_subject_oversampler import WithinSubjectOversampler

import torch
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabaz_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import scipy.ndimage
import numpy as np
import pickle
import os
import pandas as pd
from copy import copy


# TODO: run multi-threaded: disabled for the moment so there is some reproducibility in results
# TODO: fix - there were changes that need to be compacted here - so that this still works as a feature constructor
class BoTWFeatureConstructor(FeatureConstructor):
    """
    Bag of Temporal Words:
    """

    def __init__(self, parameters):
        super().__init__(parameters)
        print("Bag of Temporal Words being initiated...")

        self.codebook_name = None
        self.all_codebooks_dir = utils.create_dir(utils.get_root_path("saved_objects") + "/codebooks")

        self.codebook_plot_path = utils.get_root_path("Results") + "/" + parameters.study_name + "/codebook plots/"
        self.codebook_plots = utils.create_dir(self.codebook_plot_path)
        self.features = [self.compute_histogram]

    # TODO: make the way the axis is extracted more general
    def compute_histogram(self, cat, feature_axis):
        """
        For each interval of each chunk, the number of keypoints that belong to each cluster is calculated,
        and the distribution over all the clusters for each interval is converted to a 1D feature vector.

        Args:
            cat: the category over which to create features in the form form of distributions over clusters

        Returns:
            category_dist (ndarray): a 3D ndarray containing distribution of keypoints over different clusters. The
            shape is (nubmer of chunks, instances per chunk, number of clusters(=number of final attributes)).

        """
        # load the learning
        assert self.codebook_name is not None
        codebook_path = self._get_codebook_path(self.codebook_name)

        kmeans = pickle.load(open(codebook_path, 'rb'))

        new_chunk_list = []
        for i in range(0, cat.shape[0]):
            chunk_distributions = []
            for j in range(0, cat.shape[1]):
                interval = cat[i, j, :, :]
                interval_descriptors_list = self._produce_interval_descriptors(interval)

                if len(interval_descriptors_list) > 0:
                    interval_descriptors = np.concatenate(interval_descriptors_list, axis=0)
                    prediction = kmeans.predict(interval_descriptors)
                    interval_distribution = self._compute_interval_distribution(prediction, kmeans.n_clusters)
                else:
                    interval_distribution = np.zeros((1, kmeans.n_clusters))

                chunk_distributions.append(interval_distribution)
            new_chunk = np.concatenate(chunk_distributions, axis=0)
            new_chunk_list.append(new_chunk)
        category_dist = np.stack(new_chunk_list, axis=0)
        return category_dist

    def _compute_interval_distribution(self, prediction, num_clusters):
        """

        Args:
            prediction (ndarray): a ndarray whose elements contain the indices of the cluster centers to which a
                                  descriptor belongs
            num_clusters (int): the total number of clusters in the KMeans algorithm

        Returns:
            dist(ndarray): a ndarray of dimension (1, num_clusters) where each index represents the particular
                            cluster and its value, the number of descriptors in that interval that belong to that
                            cluster.
        """
        dist = np.zeros((1, num_clusters))
        for i in range(0, prediction.shape[0]):
            for x in np.nditer(prediction[i]):
                dist[0, x] = dist[0, x] + 1

        return dist

    def generate_codebook(self, subj_dataset, dataset_desc_name, kmeans_nclusters):

        """

        Args:
            subj_dataset:

        Returns:

        """
        dataset_desc_path = self._get_dataset_desc_path(dataset_desc_name)
        if not os.path.exists(dataset_desc_path):
            dataset_desc = self._produce_dataset_descriptors(subj_dataset, dataset_desc_name)
        else:
            dataset_desc = np.load(dataset_desc_path)

        codebook_name = dataset_desc_name + "_kmeans_" + str(kmeans_nclusters)
        codebook_path = self._get_codebook_path(codebook_name)
        if not os.path.exists(codebook_path):
            kmeans = KMeans(n_clusters=kmeans_nclusters).fit(dataset_desc)
            # save the learning to disk
            pickle.dump(kmeans, open(codebook_path, 'wb'))
        else:
            print("Codebook: ", codebook_name, "already exists in ", self.all_codebooks_dir)

        self.codebook_name = codebook_name

    def _produce_dataset_descriptors(self, subj_dataset, dataset_desc_name):
        processed_dataset = self._process_dataset(subj_dataset)

        cat_desc_list = []
        for subj_name, subj in processed_dataset.items():
            cat_data = subj.data
            cat_names = subj.categories
            for i, cat in enumerate(cat_data):
                cat_desc = self.produce_category_descriptors(cat)
                cat_name = cat_names[i]
                print("Obtained descriptors of category ", str(i), ": ", cat_name)

                if cat_desc is not None:
                    cat_desc_list.append(cat_desc)

        dataset_desc = np.concatenate(cat_desc_list, axis=0)
        dataset_desc_path = self._get_dataset_desc_path(dataset_desc_name)

        # save the numpy array containing the dataset description
        np.save(dataset_desc_path, dataset_desc)
        return dataset_desc

    def plot_kmeans(self, dataset_desc_name, kmeans_name):
        """

        Args:
            dataset_desc_name(str): The name of the serialized dataset descriptors
            kmeans_name: The name of the serialized kmeans algorithm

        """
        codebook_path = self._get_codebook_path(kmeans_name)
        dataset_desc_path = self._get_dataset_desc_path(dataset_desc_name)

        kmeans = pickle.load(open(codebook_path, 'rb'))
        dataset_desc = np.load(dataset_desc_path)
        # reduce dimensionality of points to cluster (to 2D)
        pca_ncomponents = 2
        pca = PCA(n_components=pca_ncomponents)
        data2D = pca.fit_transform(dataset_desc)
        plt.scatter(data2D[:, 0], data2D[:, 1])

        # plot the cluster centers
        # plt.hold(True)
        centers2D = pca.transform(kmeans.cluster_centers_)

        plt.scatter(centers2D[:, 0], centers2D[:, 1], marker='x', s=200, linewidths=3, c='r')
        plt.savefig(self.codebook_plot_path + dataset_desc_name + str(kmeans_name) + "_" + str(pca_ncomponents) +
                    "D.png")
        plt.show()
        return

    def score_kmeans(self, dataset_desc_name, kmeans_name):
        """
        Args:
            dataset_desc_name: The name of the serialized dataset descriptors
            kmeans_name: The name of the serialized kmeans algorithm

        Returns: a tuple containing the silhouette score, which is in the range -1 to +1, followed by the
            calinski-harabaz score

        """
        codebook_path = self._get_codebook_path(kmeans_name)
        dataset_desc_path = self._get_dataset_desc_path(dataset_desc_name)

        kmeans = pickle.load(open(codebook_path, 'rb'))
        dataset_desc = np.load(dataset_desc_path)

        labels = kmeans.labels_
        silhouette = silhouette_score(dataset_desc, labels, metric='euclidean')
        calinski_harabaz = calinski_harabaz_score(dataset_desc, labels)
        print("Silouhette score: ", silhouette)
        print("Calinsky Harabes score: ", category_balancer)
        return silhouette, calinski_harabaz

    def log_kmeans_score(self, nclusters_list, interval_size, name_extra_str=""):

        filepath = utils.get_root_path("Results") + "/" + parameters.study_name + "/codebook results/"
        filepath = utils.create_dir(filepath) + "cluster_eval.txt"

        with open(filepath, 'a') as the_file:
            for nclust in nclusters_list:
                codebook_alg_name = "_kmeans_" + str(nclust)
                dataset_desc_name = "CTS_firm_chunk_" + str(parameters.samples_per_chunk) + "_interval_" + str(
                    interval_size) + name_extra_str
                feature_constructor.generate_codebook(subject_dict, dataset_desc_name, nclust)
                silouhette, calinski_harabaz = feature_constructor.score_kmeans(dataset_desc_name, dataset_desc_name +
                                                                                codebook_alg_name)
                the_file.write(
                    'Number of clusters: ' + str(nclust) + "; Interval size: " + str(interval_size) + ":  Silouhette "
                                                                                                      "Score: " + str(
                        silouhette) + ";  Calinksi-Harabaz Score: " + str(calinski_harabaz) + "\n")

    def compute_dataset_stats(self, feature_dataset, filename):
        # create a new feature dataset that is the same as the first, except it does not have dimensions that are 1.
        squeezed_feature_dataset = {}
        for subj_name, subj in feature_dataset.items():
            new_data = np.squeeze(subj.data)
            new_subj = copy(subj)
            new_subj.data = new_data

            squeezed_feature_dataset[subj_name] = new_subj

        dataframe_dict = self.__create_dataframe_dict(squeezed_feature_dataset)
        all_subj_dataframe = self.__create_allsubj_dataframe(dataframe_dict)

        groups = all_subj_dataframe.groupby("category")
        for name, group in groups:
            l1norm = np.linalg.norm(group[[0, 1, 2, 3, 4]].values.astype(float), axis=1)
            print(":/")
        # group_desc = groups.describe()
        group_cov = groups.cov()
        print(group_cov)

        filepath = utils.get_root_path("Results") + "/" + parameters.study_name + "/codebook results/"
        file_name = utils.create_dir(filepath) + "dataset_covariance_" + filename + ".csv"
        group_cov.to_csv(file_name, sep='\t')

        # corr_aggr = corr.aggregate()
        # seaborn scatter matrix
        # g = sns.pairplot(all_subj_dataframe, hue='category', diag_kind='hist')
        # plt.show()

        # pandas correlation matrix
        # plt.matshow(all_subj_dataframe.corr())
        # plt.show()

        return all_subj_dataframe

    def __create_dataframe_dict(self, subject_dict):
        dataframe_dict = {}
        for subj_name, subj in subject_dict.items():
            dataframe_dict[subj_name] = self.__compact_subj_dataframe(subj)
        return dataframe_dict

    # this function puts the data from one subject in one dataframe (adding a column corresponding to category)
    def __compact_subj_dataframe(self, subject):
        subj_data = subject.data
        category_names = subject.categories

        category_list = []

        # convert the data from each category to pandas dataframe, as expected from seaborn
        for i, cat_data in enumerate(subj_data):
            subj_cat_data = pd.DataFrame(data=cat_data)

            # create a list of categories to append to the dataframe (so we can plot by it)
            category_name_list = [category_names[i]] * cat_data.shape[0]
            subj_cat_data['category'] = category_name_list

            category_list.append(subj_cat_data)

        compacted_dataframe = pd.concat(category_list)
        return compacted_dataframe

    def __create_allsubj_dataframe(self, dataframe_dict):
        # create a pandas dataframe that has all the subject and categories out of the dictionary that has the
        # subject name as key and its dataframe as a value
        subject_dataframe_list = []
        for subj_name, subject_dataframe in dataframe_dict.items():
            subject_dataframe['subj_name'] = subj_name
            subject_dataframe_list.append(subject_dataframe)

        allsubj_dataframe = pd.concat(subject_dataframe_list)
        return allsubj_dataframe

    def _get_codebook_path(self, codebook_name):
        """
        Returns the path to the codebook whose name is passed as an argument
        Args:
            codebook_name: The name of the codebook whose path is to be returned

        Returns:
            codebook_path: the absolute path to that codebook

        """
        codebook_path = os.path.abspath(os.path.join(self.all_codebooks_dir, codebook_name + ".sav"))
        # assert os.path.exists(codebook_path)
        return codebook_path

    def _get_dataset_desc_path(self, dataset_desc_name):
        """
        Returns the path to the numpy array containing the dataset description whose name is passed as an
        argument

        Args:
        dataset_desc_name: The name of the dataset descriptors whose path is to be returned

        Returns:
            dataset_desc_path: the absolute path to that numpy array containing dataset descriptors
        """
        dataset_desc_path = os.path.abspath(os.path.join(self.all_codebooks_dir, dataset_desc_name + ".npy"))
        return dataset_desc_path

    def produce_category_descriptors(self, cat):
        """

        Args:
            cat (ndarray): category to be described

        Returns:
            cat_descriptors:

        """
        # cat = torch.from_numpy(cat).cuda(async=True)
        descriptor_list = []
        for i in range(0, cat.shape[0]):
            for j in range(0, cat.shape[1]):
                interval = cat[i, j, :, :]
                interval_desc_list = self._produce_interval_descriptors(interval)
                descriptor_list.append(interval_desc_list)

        descriptor_list = [desc for sublist in descriptor_list for desc in sublist]
        cat_descriptors = np.concatenate(descriptor_list, axis=0)
        return cat_descriptors

    def _produce_interval_descriptors(self, interval):
        """

        Args:
            interval: a 2D interval over which to find keypoints and produce their descriptros

        Returns:
            descriptor_list (list): list of descriptors of the interval keypoints

        """
        descriptor_list = []
        octave = self._create_octave(interval)
        diff_of_gaussian = self._compute_diff_of_gaussian(octave)
        keypoints = self._find_keypoints(diff_of_gaussian)

        keypoint_desc = self._describe_keypoints(octave, keypoint_list=keypoints)
        if keypoint_desc is not None:
            descriptor_list.append(keypoint_desc)

        return descriptor_list

    def _create_octave(self, interval):
        """
        Blurs each column of the input interval by repeated Gaussian filtering at different scales.

        Args:
            interval: the signal whose columns are to be individually filtered

        Returns:
            octave (list): a list of intervals (ndarrays) filtered at different scales (2D)- each separately
                filtered

        """
        k = math.sqrt(2)
        sigma = 0.5

        octave = []
        for i in range(0, 5):
            sigma = k * sigma
            smoothed_interval = self._smooth_interval(interval, sigma)
            octave.append(smoothed_interval)

        return octave

    def _smooth_interval(self, interval, sigma):
        """
        Each column of the input interval is smoothed at a particular scale.

        Args:
            interval: the signal whose columns are to be individually filtered
            sigma: smoothing scale

        Returns:
            smoothed_interval (ndarray): filtered 2D array at sigma scale

        """
        signal_list = []
        for i in range(0, interval.shape[-1]):
            signal = interval[:, i]
            filtered = scipy.ndimage.gaussian_filter1d(input=signal, sigma=sigma)
            signal_list.append(filtered)

        smoothed_interval = np.stack(signal_list, axis=1)
        return smoothed_interval

    def _compute_diff_of_gaussian(self, octave):
        """
        Computes differences of consecutive blurred signals passed in the octave list

        Args:
            octave (list of ndarray): list of progressively blurred intervals (each column individually)

        Returns:
            diff_of_gaussian_list (list of ndarray): a list of differences between consecutive input intervals

        """

        diff_of_gaussian_list = []
        for i in range(1, len(octave)):
            diff_of_gaussian = np.subtract(octave[i], octave[i - 1])
            diff_of_gaussian_list.append(diff_of_gaussian)
        return diff_of_gaussian_list

    def _find_keypoints(self, octave_dog):
        """
        Finds points in the interval that are either the maximum or minimum of their neighbourhood within at least
        one dimension.

        Args:
            octave_dog: a list of difference of gaussian intervals

        Returns:
            signal_keypoints (list): a list of tuples - each tuple contains coordinates for a discontinuity.
                                     The first coordinate indicates the scale, while the second, the time point.
        """
        num_signal = octave_dog[0].shape[-1]

        signal_keypoints = []
        for i in range(0, num_signal):
            signal_list = []
            for interval in octave_dog:
                signal = interval[:, i]
                signal_list.append(signal)
            keypoints_1d = self._find_keypoints_1d(signal_list)
            signal_keypoints.append(keypoints_1d)

        all_keypoints = [coords for sublist in signal_keypoints for coords in sublist]
        return all_keypoints

    # admit keypoint if DoG point is smaller or larger than all the points in its neighbourhood
    def _find_keypoints_1d(self, signals_1d):
        """

        Args:
            signals_1d:

        Returns:

        """
        signals_1d_list = []
        for ndarray in signals_1d:
            signals_1d_list.append(ndarray.tolist())

        keypoint_idx = []
        for i in range(1, len(signals_1d_list) - 1):
            prev_signal = signals_1d_list[i - 1]
            signal = signals_1d_list[i]
            next_signal = signals_1d_list[i + 1]

            for j in range(1, len(signal) - 1):
                if (signal[j] > prev_signal[j] and signal[j] > next_signal[j] and
                    signal[j] > signal[j - 1] and signal[j] > signal[j + 1] and
                    signal[j] > prev_signal[j - 1] and signal[j] > prev_signal[j + 1] and
                    signal[j] > next_signal[j - 1] and signal[j] > next_signal[j + 1]) or \
                        (signal[j] < prev_signal[j] and signal[j] < next_signal[j] and
                         signal[j] < signal[j - 1] and signal[j] < signal[j + 1] and
                         signal[j] < prev_signal[j - 1] and signal[j] < prev_signal[j + 1] and
                         signal[j] < next_signal[j - 1] and signal[j] < next_signal[j + 1]):
                    # print("Found keypoint in signal ", i, "at location ", j, "!")
                    keypoint_idx.append((i, j))
        return keypoint_idx

    def _describe_keypoints(self, octave, keypoint_list=None):
        """
        Describes every point of an interval by default, unless there is a keypoint_list passed as an argument,
        in which case it uses that to find the indices of points to describe. 

        Args:
            octave (list): a list of arrays representing the same signal progressively blurred.
            keypoint_list (list): a list of tuples indicating extrama in the signal scale-space. The first element
                                  of the tuple is the level of blur (scale), - indicating which of the octaves to
                                  select from the list - while the second element of the tuple stands for the time
                                  position within that octave that is a maximum or minimum.

        Returns:
            processed_octave (ndarray): a ndarray of 2*nb*number_of_blurred_images feature vector that describes
                each point of the interval

        """
        nb = 4
        a = 4

        if keypoint_list is None:
            processed_signals = []
            for filtered_signal in octave:
                signal_desc_list = []
                for i in range(0, octave[0].shape[-1]):
                    desc = self._describe_signal_1d(filtered_signal[:, i], nb, a)
                    signal_desc_list.append(desc)
                signal_desc = np.concatenate(signal_desc_list, axis=1)
                processed_signals.append(signal_desc)

            result = np.concatenate(processed_signals, axis=1)

        else:
            if len(keypoint_list) > 0:
                keypoint_descriptors = []
                keypoint_descriptors_2D = []
                for i, keypoint in enumerate(keypoint_list):

                    # print("keypoint: ", keypoint)
                    scale = keypoint[0]
                    interval = octave[scale]

                    point_idx = keypoint[1]
                    column_desc = []
                    neighborhood_2D_list = []
                    for j in range(0, interval.shape[-1]):
                        signal = interval[:, j]
                        keypoint_neighbourhood = self._get_neighbourhood(signal, point_idx, nb, a)
                        neighborhood_2D_list.append(keypoint_neighbourhood)

                        descriptor = self._describe_each_point_1D(keypoint_neighbourhood, nb, a)
                        column_desc.append(descriptor)
                    all_column_desc = np.concatenate(column_desc, axis=0)
                    keypoint_descriptors.append(all_column_desc)

                    neighborhood_2D = np.stack(neighborhood_2D_list, axis=1)
                    descriptors_2D = self._describe_each_point_2D(neighborhood_2D, nb, a)
                    keypoint_descriptors_2D.append(descriptors_2D)

                all_descriptors_2D = np.stack(keypoint_descriptors_2D, axis=0)
                all_descriptors_1D = np.stack(keypoint_descriptors, axis=0)
            else:
                all_descriptors_1D = None
                all_descriptors_2D = None

            return all_descriptors_2D

    def _get_neighbourhood(self, signal_1d, point_idx, nb, a):
        """
        Constructs a neighbourhood around a point to be described, with nb*a/2 points before and after that point.

        Args:
            signal_1d (ndarray): one column of an interval
            point_idx (int): the index of the point in the interval to be described (whose neighbourhood to get)
            nb: number of blocks to describe the point
            a: number of points in each block

        Returns:
            keypoint_neighbourhood (ndarray): a 1D array that contains the neighbourhood with the point to be
                                              described in center. The size of the array is nb * a + 1.

        """
        assert len(signal_1d) > 0

        start = int(point_idx - (nb * a) / 2)
        stop = int(point_idx + (nb * a) / 2 + 1)

        # if there aren't enough values to form blocks ahead of the keypoint, repeat the first value
        if point_idx < nb * a / 2:
            pad_n = nb * a / 2 - point_idx
            padding = np.repeat(signal_1d[0], pad_n)
            signal = signal_1d[0: stop]

            keypoint_neighbourhood = np.concatenate((padding, signal), axis=0)

        # if there aren't enough values to form blocks after the keypoint, repeat the last value
        elif signal_1d.shape[0] < stop:
            signal = signal_1d[start: stop]

            pad_n = stop - signal_1d.shape[0]
            padding = np.repeat(signal_1d[-1], pad_n)
            keypoint_neighbourhood = np.concatenate((signal, padding), axis=0)

        else:
            keypoint_neighbourhood = signal_1d[start: stop]

        return keypoint_neighbourhood

    def _describe_signal_1d(self, signal_1d, nb, a):
        """
        Describes each point of the input signal in terms of positive and negative gradients in its neighbourhood.

        Args:
            signal_1d (ndarray): a 1-dimensional signal, each point of which will be described in terms of gradients
                of its surrounding blocks
            nb: the total number of blocks to consider per point (half on each side) where positive and negative
                gradients' sums will be computed
            a: the number of points in each block

        Returns (ndarray): a numpy array of all points where each is described by 2 values per block (2*nb feature
            vector)

        """
        assert nb % 2 == 0, "The number of blocks that describe the keypoint needs to be even, so we can get an " \
                            "equal number of points before and after the keypoint."

        keypoint_descriptors = []
        for pos, point in enumerate(signal_1d):
            keypoint_neighbourhood = self._get_neighbourhood(signal_1d, pos, nb, a)
            descriptor = self._describe_each_point_1D(keypoint_neighbourhood, nb, a)

            keypoint_descriptors.append(descriptor)

        signal_desc = np.stack(keypoint_descriptors, axis=0)
        return signal_desc

    def _describe_each_point_2D(self, keypoint_neighborhood_2D, nb, a):
        assert keypoint_neighborhood_2D is not None, "No neighbourhood has been assigned to the keypoint."
        assert keypoint_neighborhood_2D.shape[0] == nb * a + 1

        # 2 2-D arrays are returned: first standing for gradients in rows, and second for gradients in columns
        gradient_rows, gradient_cols = np.gradient(keypoint_neighborhood_2D)
        filtered_gradient_rows = scipy.ndimage.gaussian_filter(input=gradient_rows, sigma=nb * a / 2)
        filtered_gradient_cols = scipy.ndimage.gaussian_filter(input=gradient_cols, sigma=nb * a / 2)

        point_idx = int(nb * a / 2)

        all_gradients = []
        for i in range(0, filtered_gradient_rows.shape[1]):
            filtered_gradient_rows_1D = filtered_gradient_rows[:, i]
            blocks = self._create_blocks(filtered_gradient_rows_1D, point_idx, a)
            gradients_rows_1D = self._get_block_gradients(blocks)
            all_gradients.append(gradients_rows_1D)

        for i in range(0, filtered_gradient_cols.shape[1]):
            filtered_gradient_cols_1D = filtered_gradient_cols[:, i]
            blocks = self._create_blocks(filtered_gradient_cols_1D, point_idx, a)
            gradients_cols_1D = self._get_block_gradients(blocks)
            all_gradients.append(gradients_cols_1D)

        all_gradient_sums = np.concatenate(all_gradients)
        return all_gradient_sums

    def _describe_each_point_1D(self, keypoint_neighbourhood, nb, a):
        """
        Each keypoint is described in terms of the sum of positive and negative gradients of blocks of other points
        around it. The interval keypoint_neighbourhood, whose midpoint is the keypoint being characterized,
        is first filtered with a gaussian filter of scale nb*a/2 to weigh the importance of points according to
        proximity with the keypoint.

        Args:
            keypoint_neighbourhood:
            nb: the total number of blocks around a keypoint
            a: the number of points in one block

        Returns:
            all_gradient_sums (ndarray): a 2*nb long ndarray where from each block, the sum of positive gradients,
            and the sum of negative gradients are maintained

        """
        assert keypoint_neighbourhood is not None, "No neighbourhood has been assigned to the keypoint."
        assert keypoint_neighbourhood.shape[0] == nb * a + 1

        gradient = np.gradient(keypoint_neighbourhood)
        filtered_gradient = scipy.ndimage.gaussian_filter1d(input=gradient, sigma=nb * a / 2)
        point_idx = int(nb * a / 2)

        blocks = self._create_blocks(filtered_gradient, point_idx, a)
        gradient_sums = self._get_block_gradients(blocks)

        return gradient_sums

    def _create_blocks(self, filtered_gradient_1D, point_idx, a):
        blocks = []
        for i in range(0, point_idx, a):
            block = filtered_gradient_1D[i:i + a]

            blocks.append(block)

        for i in range(point_idx + 1, filtered_gradient_1D.shape[0], a):
            block = filtered_gradient_1D[i:i + a]
            blocks.append(block)

        return blocks

    def _get_block_gradients(self, blocks):
        all_gradients = []
        for j, block in enumerate(blocks):
            pos = 0
            neg = 0
            for point in block:
                if point < 0:
                    neg = neg + point
                if point > 0:
                    pos = pos + point
            all_gradients.append([pos, neg])

        gradient_sums = np.array(all_gradients).flatten()
        return gradient_sums


if __name__ == "__main__":
    print("Running feature_constructor module...")
    print("Is cuda available?", torch.cuda.is_available())

    config_dir = "config_files"
    config = StudyConfig(config_dir)

    # create a template of a configuration file with all the fields initialized to None
    config.create_config_file_template()
    parameters = config.populate_study_parameters("CTS_5taps_per_button.toml")

    # generating the data from files
    data = DataConstructor(parameters)
    subject_dict = data.get_subject_dataset()

    # define a category balancer (implementing the abstract CategoryBalancer)
    category_balancer = WithinSubjectOversampler()
    dataset_processor = StatDatasetProcessor(parameters, balancer=category_balancer)

    feature_constructor = BoTWFeatureConstructor(dataset_processor, parameters, feature_axis=2)
    dataset_desc_name = "CTS_firm_chunk_" + str(parameters.samples_per_chunk) + "_interval_" + str(
        parameters.feature_window) + "_test"

    nclusters_list = [5]
    # feature_constructor.log_kmeans_score(nclusters_list, parameters.feature_window, "_old_alg")
    # feature_constructor.log_kmeans_score(nclusters_list, interval_size_list)
    feature_constructor.generate_codebook(subject_dict, dataset_desc_name, 30)

    feature_dataset = feature_constructor.produce_feature_dataset(subject_dict)
    file_desc_name = parameters.study_name + "_interval_" + str(parameters.feature_window) + "_old_alg_kmeans_" + str(
        nclusters_list[0])

    dataset_corr = feature_constructor.compute_dataset_stats(feature_dataset, file_desc_name)
    print("Done")
