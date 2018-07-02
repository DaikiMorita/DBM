import unittest
import os
from DBM.src.python3.Manager import ExFileHandler
import sys
import numpy as np

sys.path.append("DBM.test.ManagerTest")


class TestExFileManager(unittest.TestCase):
    """
    unit test of features in ExFileManager
    """

    def setUp(self):
        """
        procedures before every tests are started. This code block is executed only once.
        :return:
        """

        self.ex_file_handler = ExFileHandler.ExFileHandler()

    def test_count_empty_file(self):
        """
        test of count empty file
        :return:
        """

        expected = 2
        actual = self.ex_file_handler.count_empty_file("./test_data/test")

        self.assertEqual(expected, actual)

    def test_count_up_data_num(self):
        """
        test of count up date num
        :return:
        """

        expected = 3
        actual = self.ex_file_handler.count_up_data_num("./test_data/test")

        self.assertEqual(expected, actual)

    def test_normalized_image_array(self):
        """
        test of normalized image array
        :return:
        """
        expected = [1.0] * 2500
        actual = self.ex_file_handler.normalized_img_list(os.path.join("./test_data/test_img", "white"))
        self.assertEqual(expected, actual)

    def test_get_data_dir(self):
        """
        test of get data in dir
        :return:
        """

        expected = [[0.0] * 2500, [1.0] * 2500]
        actual = self.ex_file_handler.get_data_in_dir("./test_data/test_img")
        self.assertEqual(expected, actual)

    def test_get_image_width_height(self):
        """
        test get image width height
        :return:
        """
        expected = (50, 50)
        actual = self.ex_file_handler.get_image_width_height(os.path.join("./test_data/test_img", "black"))
        self.assertEqual(expected, actual)

    def test_write_to_file(self):
        """
        test write to file
        :return:
        """

        self.ex_file_handler.write_to_file("Hello!!", "./test_data/hello")
        expected = True

        actual = os.path.isfile(os.path.join("./test_data", "hello"))
        self.assertEqual(expected, actual)

    def test_np_arr_save(self):
        """
        test_numpy_array_save
        :return:
        """

        self.ex_file_handler.np_arr_save("./test_data", "test", np.array([1, 2, 3]))
        expected = True
        actual = os.path.isfile(os.path.join("./test_data", "test.npy"))
        self.assertEqual(expected, actual)

    def test_read_img_all_data_labels(self):
        """
        test of read_image_data
        :return:
        """

        labels, data = self.ex_file_handler.read_img_all_data_labels(os.path.join("./test_data", "img_dirs"))

        labels.sort()
        actual_1 = labels
        actual_2 = data

        expected_1 = ["A", "A", "B", "B", "C", "C"]
        expected_2 = [[0.0] * 2500, [1.0] * 2500, [0.0] * 2500, [1.0] * 2500, [0.0] * 2500, [1.0] * 2500]

        self.assertEqual(expected_1, actual_1)
        self.assertEqual(expected_2, actual_2)
