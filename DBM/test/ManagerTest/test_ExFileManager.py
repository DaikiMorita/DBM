import unittest
from DBM.src.python3.Manager import ExFileManager


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
