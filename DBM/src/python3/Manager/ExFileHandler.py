# coding=utf-8


import configparser
import os
import tqdm
import numpy as np
from PIL import Image


class ExFileHandler(object):
    """
    Manages external files
    """

    def __init__(self):

        # Reads .config.ini
        ini_file = configparser.ConfigParser()

    def read_img_all_data_labels(self, path_all_dirs):
        """
        Reads data.
        This method can be applied especially when you try to read "image"s.
        :return: num_all_data, formated_data, each_label_data
        """

        all_data = []
        all_labels = []

        for dir in os.listdir(path_all_dirs):
            path_to_data = os.path.join(path_all_dirs, dir)

            labels = [dir] * self.count_up_data_num(path_to_data)
            data = self.get_data_in_dir(path_to_data)

            all_labels.append(labels)
            all_data.append(data)

        # all_data_array above was organized like [[data with label A],[data with label B]....]
        # a format like [data with label A, data with label B,...] is easy to use.
        # For example, for shuffling or making mini-batch.
        flattened_labels = []
        flattened_data = []
        for labels, data in zip(all_labels, all_data):

            for label in labels:
                flattened_labels.append(label)

            for datum in data:
                flattened_data.append(datum)

        return flattened_labels, flattened_data

    def get_data_in_dir(self, path_to_data_dir):
        """
        Gets all data in the all dirs which exist in the specified dir.
        :param path_to_data_dir: path to the dir where all dirs with data exist
        :return: 1st: a float scalar meaning the total amount of data
                 2nd: numpy array where all data are stored. Each row corresponds to an data
        """

        data_array = []

        for data in tqdm.tqdm(os.listdir(path_to_data_dir)):

            path_to_data = os.path.join(path_to_data_dir, data)

            if os.stat(path_to_data).st_size != 0:
                data_array.append(self.normalized_img_list(path_to_data))

        return data_array

    def normalized_img_list(self, path_to_image):

        img = Image.open(path_to_image)
        width, height = img.size
        return [img.getpixel((i, j)) / 255 for j in range(height) for i in range(width)]

    def count_up_data_num(self, dir):
        """
        Counts up the number of non-0-size files in a folder
        :param dir: folder name where you want to know the amount of files
        :return: the number of files in the folder
        """
        return len(os.listdir(dir)) - self.count_empty_file(dir)

    def count_empty_file(self, dir):
        """
        Counts up the number of empty files.
        :param dir: the folder where files of which you want to know the amount exist.
        :return: amount
        """

        return len([index for index, file in enumerate(os.listdir(dir)) if
                    os.stat(os.path.join(dir, file)).st_size == 0])

    def get_image_width_height(self, path_to_image):
        """
        get width and height of an image
        :param path_to_image:
        :return:
        """

        img = Image.open(path_to_image)
        width, height = img.size

        return width, height

    def write_to_file(self, data, filename):
        """

        :param filename:
        :param data:
        :return:
        """

        with open(filename, mode='a', encoding='utf-8') as fh:
            fh.write('%s\n' % data)

    def np_arr_save(self, path, filename, array):
        """
        Saves numpy array into a directory.
        :param path: path
        :param filename: filename
        :param array: array to be saved
        :return: None
        """

        np.save('%s.npy' % os.path.join(path, filename), array)
