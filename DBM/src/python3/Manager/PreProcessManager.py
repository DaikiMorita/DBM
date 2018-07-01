# coding=utf-8

import numpy as np
import os
import DBM.src.Viewer.Viewer as v
import shutil


class PreProcessManager(object):
    """
    in charge of pre processes.
    """

    def __init__(self):
        self.viewer = v.Viewer()

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
