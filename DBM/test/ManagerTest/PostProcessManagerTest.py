import unittest
import numpy as np
import os
import numpy.testing as npt
from DBM.src.Manager import PostProcessManager


class PostProcessManagerTest(unittest.TestCase):
    """
    unit test of features in PostProcessManager
    """

    def setUp(self):
        """
        procedures before every tests are started. This code block is executed only once.
        :return:
        """
        self.post_process_manager = PostProcessManager.PostProcessManager()

    def test_save_numpy_array(self):
        """
        test of save_numpy_array
        check whether a file with '.npy' is created at a specified path Even with your naked eye
        :return:
        """

        arr = np.ones((100, 100))
        filename = "test_save_numpy_array"
        path = ""

        self.post_process_manager.save_numpy_array(arr, filename, path)

        expected = True
        actual = os.path.isfile(os.path.join(path, filename + ".npy"))
        self.assertEqual(expected, actual)
