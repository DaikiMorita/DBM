import unittest
import numpy as np
import numpy.testing as npt
from DBM.src.python3.Manager import DataProcessor
import os


class DataProcessManagerTest(unittest.TestCase):
    """
    unit test of features in PreProcessManager
    """

    def setUp(self):
        """
        procedures before every tests are started. This code block is executed only once.
        :return:
        """
        self.data_process_manager = DataProcessor.DataProcessor()

    def test_z_score_normalization(self):
        """
        test of z-score normalization to 2-d numpy array
        :return: None
        """

        # array normalized into z-score
        arr = np.array([[0, 2, 2, 4], [0, 2, 2, 4]])

        expected = np.array([[-np.sqrt(2), 0, 0, np.sqrt(2)], [-np.sqrt(2), 0, 0, np.sqrt(2)], ])

        actual = self.data_process_manager.z_score_normalization(arr)

        npt.assert_array_almost_equal(expected, actual)

    def test_make_mini_batch(self):
        """
        test of making mini batches from 2-d list
        :return: None
        """

        # 2-d list
        data_list = [[1, 1, 1],
                     [2, 2, 2],
                     [3, 3, 3],
                     [4, 4, 4],
                     [5, 5, 5],
                     [6, 6, 6],
                     [7, 7, 7],
                     [8, 8, 8],
                     [9, 9, 9],
                     [10, 10, 10],
                     ]
        # makes mini batches with 3 elements
        expected = [[[1, 1, 1], [2, 2, 2], [3, 3, 3]], [[4, 4, 4], [5, 5, 5], [6, 6, 6]],
                    [[7, 7, 7], [8, 8, 8], [9, 9, 9]], [[10, 10, 10], [1, 1, 1], [2, 2, 2]]]

        actual = self.data_process_manager.make_mini_batch(data_list, 3)

        self.assertEqual(expected, actual)

    def test_save_numpy_array(self):
        """
        test of save_numpy_array
        check whether a file with '.npy' is created at a specified path Even with your naked eye
        :return:
        """

        arr = np.ones((100, 100))
        filename = "test_save_numpy_array"
        path = ""

        self.data_process_manager.save_numpy_array(arr, filename, path)

        expected = True
        actual = os.path.isfile(os.path.join(path, filename + ".npy"))
        self.assertEqual(expected, actual)
