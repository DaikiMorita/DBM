import os
import shutil
import numpy as np


class DataProcessManager(object):
    """
    This class provides various methods to process data
    """

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

    def make_dir(self, path, dir_name):
        """
        mkdir
        :param dir_name:
        :param path:
        :return:
        """

        # To Do: Error msg should be written in log file, not direct console printing.
        try:
            os.mkdir(os.path.join(path, dir_name))
        except FileExistsError:
            pass

    def copy_file(self, before_path, after_path):
        """
        copy file into path
        :param before_path:
        :param after_path:
        :return:
        """
        # To Do: Error msg should be written in log file, not direct console printing.
        try:
            shutil.copy2(before_path, after_path)
        except FileExistsError:
            pass

    def save_numpy_array(self, array, filename, path=''):
        """

        :param array: numpy array
        :param filename: name for saving
        :param path: path where array will be saved
        :return: None
        """
        # To Do: Error msg should be written in log file, not direct console printing.
        try:
            np.save('%s.npy' % os.path.join(path, filename), array)
        except FileNotFoundError:
            pass
