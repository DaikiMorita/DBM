import unittest
import numpy as np
import numpy.testing as npt
from DBM.src.Manager import ExFileManager


class TestExFileManager(unittest.TestCase):
    """
    unit test of features in ExFileManager
    """

    def setUp(self):
        """
        procedures before every tests are started. This code block is executed only once.
        :return:
        """

        ex_file_manager = ExFileManager.ExFileManager()
