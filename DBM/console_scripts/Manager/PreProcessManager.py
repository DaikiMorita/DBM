# coding=utf-8

import collections
import configparser
import numpy as np
import random
import DBM.console_scripts.Viewer.Viewer as v


class PreProcessManager(object):
    """
    in charge of pre processes.
    """

    def __init__(self):
        self.viewer = v.Viewer()

    def z_score_normalization(self, data_array):
        """
        Normalizes data (the data is transformed into the one with mean 0 and variance 1)
        :param data_array: 1-d or 2-d data array
        :return: normalized data_array
        """

        average = np.sum(data_array, axis=1) / data_array.shape[1]

        data_minus_average = np.array([d - a for d, a in zip(data_array, average)])

        sigma = np.sqrt(np.sum(np.power(data_minus_average, 2), axis=1) / data_minus_average.shape[1])
        sigma[sigma == 0] = 0.001

        return np.array([d / s for d, s in zip(data_minus_average, sigma)])

    def make_mini_batch(self, data_list, mini_batch_size):
        """
        makes mini bathes from list-type data_array
        :param data_list: 2-d list
        :param mini_batch_size: size of mini batches
        :return:3d-list. A returned list will contain mini batches.
        Each batch will contain lists as you specified at the parameter.
        """

        # Now that data_array was shuffled,
        # mini batches will contain data with a different label at the almost same rate, statistically,
        # even when mini batches are made by extracting data from the top of the list 'data_array'

        data_array_length = len(data_list)
        rest = data_array_length % mini_batch_size

        mini_batches = [data_list[i:i + mini_batch_size] for i in
                        range(0, data_array_length - rest, mini_batch_size)]

        rest_batch = data_list[data_array_length - rest:] + data_list[0:mini_batch_size - rest]
        mini_batches.append(rest_batch)

        return mini_batches

    def decorrelation(self, data_array):

        if isinstance(data_array, np.ndarray):
            data_type = 'numpy'
        elif isinstance(data_array, list):
            data_type = 'list'
            data_array = np.array(data_array)
        else:
            self.viewer.display_message(
                "Decorrelation Error: data_array should be list or numpy array type.\n")
            raise Exception

        if data_array.ndim == 2:

            # variance-covariance matrix
            sigma = np.cov(data_array)

            # eigen-vectors
            _, eig_vectors = np.linalg.eig(sigma)

            # linear transformation
            return np.dot(eig_vectors.T, data_array.T).T


        elif data_array.ndim == 3:

            flattened_array = np.empty((1, data_array.shape[0]))
            decorrelated_array = np.empty((1, data_array.shape[0]))

            row = data_array.shape[1]
            column = data_array.shape[2]

            for index, data in enumerate(data_array):
                data = np.reshape(data, (1, row * column))

                flattened_array[index] = data

            # variance-covariance matrix
            sigma = np.cov(flattened_array)

            # eigen-vectors
            _, eig_vectors = np.linalg.eig(sigma)

            # linear transformation
            decorrelated_data = np.dot(eig_vectors.T, decorrelated_array.T).T

            return np.array([np.reshape(data, (row, column)) for data in decorrelated_array])

        else:
            self.viewer.display_message(
                "Decorrelation Error: data_array should be 2 or 3 dimension.\n")
            raise Exception

    def display_image(self):
        pass
