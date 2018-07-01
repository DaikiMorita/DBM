# coding=utf-8

import numpy as np
import os
from PIL import Image


class PostProcessManager(object):
    """
    in charge of post processes.
    """

    def __init__(self, ):
        # Reads a config file
        # self.threshold_h_1 = float(ini_file['Parameter']['threshold_h_1'])

        self.lineNotifier = LineNotifier.LineNotifier()
        self.viewer = Viewer.Viewer()

    def determine_fired_H(self, each_label_data, C, W):

        label_H = []
        for l_d in each_label_data:
            H_sum = np.zeros((1, C.shape[1]))
            for d in l_d[1]:
                H = self.softmax(C, np.array(d), W)

                H_sum += H
            H_sum = H_sum / len(l_d[1])
            H_sum[H_sum >= 0.9] = 1
            H_sum[H_sum < 0.9] = 0

            print(H_sum.tolist())
            label_H.append([l_d[0], H_sum.tolist()])

        return label_H

    def softmax(self, C, X, W):

        input_sum = (np.dot(W, X.T)).T + C

        input_sum = input_sum - np.max(input_sum)
        exp_input_sum = np.exp(input_sum)
        sum_exp_input_sum = np.sum(exp_input_sum)

        return exp_input_sum / sum_exp_input_sum

    def array_to_image(self, array, *, image_size=(), store_path="", image_name="image", extension='jpg',
                       Line=False):
        """
        Changes array into image.
        :param array: array to be changed into image
        :param image_size: size of images. valid in 1-d list or numpy array.
        :param store_path: path for staring image
        :param image_name: name of image
        :param extension: extension of image such as jpg,png...
        :param Line: if True, the image will be sent to Line
        :return: None
        """

        if isinstance(array, np.ndarray):
            pass

        elif isinstance(array, list):
            array = np.array(array)

        else:
            self.viewer.disp_msg_console(
                "array_to_image Error: data_array should be list or numpy array type.\n")

        if array.ndim == 1:
            array = np.reshape(array, image_size)
            name = '%s.%s' % (image_name, extension)
            path = os.path.join(store_path, name)
            Image.fromarray(np.uint8((array / np.max(array)) * 255)).save(path)

            if Line:
                self.lineNotifier.send_line(name)

        elif array.ndim == 2:
            name = '%s.%s' % (image_name, extension)
            path = os.path.join(store_path, name)
            Image.fromarray(np.uint8((array / np.max(array)) * 255)).save(path)

            if Line:
                self.lineNotifier.send_line(name)

        elif array.ndim == 3:

            for index, a in enumerate(array):
                name = '%s_%d.%s' % (image_name, index, extension)
                path = os.path.join(store_path, name)
                Image.fromarray(np.uint8((a / np.max(a)) * 255)).save(path)

                if Line:
                    self.lineNotifier.send_line(name)
        else:
            self.viewer.disp_msg_console(
                "array_to_image Error: array should be at most 3-d.\n")
