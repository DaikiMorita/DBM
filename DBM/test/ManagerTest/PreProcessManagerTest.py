import unittest
import numpy as np
import numpy.testing as npt
from DBM.console_scripts.Manager import PreProcessManager


class PreProcessManagerTest(unittest.TestCase):
    """
    unit test of features in PreProcessManager
    """

    def test_z_score_normalization(self):
        """
        test of z-score normalization to 2-d numpy array
        :return: None
        """
        pre_process_manager = PreProcessManager.PreProcessManager()

        # array normalized into z-score
        arr = np.array([[0, 2, 2, 4], [0, 2, 2, 4]])

        expected = np.array([[-np.sqrt(2), 0, 0, np.sqrt(2)], [-np.sqrt(2), 0, 0, np.sqrt(2)], ])

        actual = pre_process_manager.z_score_normalization(arr)

        npt.assert_array_almost_equal(expected, actual)
