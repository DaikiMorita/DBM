import configparser
from sklearn.model_selection import train_test_split
from PIL import Image
import numpy as np
from datetime import datetime
import time
import os

from python3.Manager import PreProcessManager
from python3.Manager import ExFileHandler
from python3.Manager import DataProcessor
from python3.Model import GaussianBernoulliRBM


class MainController(object):
    """
    Controls main process
    """

    def __init__(self):

        self.app_conf = configparser.ConfigParser()
        self.app_conf.read("application.ini")

        self.learn_conf = configparser.ConfigParser()
        self.learn_conf.read("learning.ini")

        self.dir_for_saving_result = os.path.join(self.app_conf["Setting"]["path_result_dir"],
                                                  datetime.now().strftime("%Y-%m-%d_%H:%M:%S"))

        # TO DO: is there any way automatically to create instances like java spring @Autowired?
        self.data_process_manager = DataProcessor.DataProcessor()
        self.pre_process_manager = PreProcessManager.PreProcessManager()
        self.ex_file_handler = ExFileHandler.ExFileHandler()

    # To Do: conduct integration test as much as possible
    def start_main_pricess(self):
        """
        Main Process:
        :return:
        """

        ##################
        # 1. Preparation #
        ##################

        # [1] makes a dir for saving results
        self.data_process_manager.make_dir("", self.dir_for_saving_result)

        # [2] copy learning.ini into a result dir
        self.data_process_manager.copy_file(os.path.join(self.app_conf["Setting"]["path_learn_ini"],
                                                         self.app_conf["Setting"]["name_learn_ini"]),
                                            self.dir_for_saving_result)

        #################
        # 2. Reads data #
        #################

        # num_all_data: float scalar, total amount of data
        # all_data_array: 2-d list (dim: num_all_data * each data dimension)
        all_labels, all_data = self.ex_file_handler.read_img_all_data_labels(self.app_conf["Setting"]["path_data_dir"])

        ##################
        # 3. Pre-process #
        ##################

        # [1] Normalizes data
        all_data = self.data_process_manager.z_score_normalization(all_data)

        data_train, data_test, label_train, label_test = train_test_split(all_data, all_labels, test_size=0.3,
                                                                          shuffle=True)

        # [2] makes mini-batches where data with different labels will be contained at the almost same rate.
        data_train_batch = self.data_process_manager.make_mini_batch(data_train,
                                                                     self.learn_conf["General"]["mini_batch_size"])
        label_train_batch = self.data_process_manager.make_mini_batch(label_train,
                                                                      self.learn_conf["General"]["mini_batch_size"])

        ###############
        # 4. Learning #
        ###############
        # C, B, W, sigma = self.CD_learning(data_train_batch)

        ###################
        # 5. Post-Process #
        ###################

        # self.post_process_manager.determine_fired_H(each_label_data, C, W)

        ###########
        # 6. Test #
        ###########

        # self.test(C, W, sigma, label_list, H_list)

        ####################
        # 7. Saves results #
        ####################

        # self.save_result(C, B, W, sigma)

        ##################
        # 8. Finished !! #
        ##################

    def CD_learning(self, mini_batch):
        """
        Contastive divergence learning
        :param dict_data_parameter: dictionary with keys for learning
        :return: C, B, W, sigma
        """

        ######################
        # General Parameters #
        ######################

        epoch = int(self.learn_conf['General']['Epoch'])

        num_visible_units = self.width * self.height
        num_hidden_units = int(self.learn_conf.get('General', 'Num_Hidden_Unit'))
        learning_rate = float(self.learn_conf.get('General', 'Learning_Rate'))

        ############
        # sampling #
        ############
        sampling_times = int(self.learn_conf.get('Special', 'Smapling_Times'))

        sampling_type = ''.join(self.learn_conf['Special']['Smapling_Type'])
        if not (sampling_type == 'CD' or sampling_type == 'PCD'):
            raise Exception

        ############
        # momentum #
        ############
        if ''.join(self.learn_conf['SpecialParameter']['Momentum']) == 'Yes':
            momentum_rate = float(self.learn_conf['Special']['Momentum_Rate'])
        else:
            momentum_rate = 0

        ################
        # weight_decay #
        ################
        if ''.join(self.learn_conf['Special']['Weight_Decay']) == 'Yes':
            weight_decay_rate = float(self.learn_conf['Special']['Weight_Decay_Rate'])
        else:
            weight_decay_rate = 0

        #########################
        # sparse_regularization #
        #########################
        if ''.join(self.learn_conf['Special']['Sparse_Regularization']) == 'Yes':
            sparse_regularization = (float(self.learn_conf['Special']['Sparse_Regularization_Target']),
                                     float(self.learn_conf['Special']['Sparse_Regularization_Rate']))

        else:
            sparse_regularization = (0, 0)

        width_sf = int(self.learn_conf['Special']['Width_Spread_Function'])
        height_sf = int(self.learn_conf['Special']['Height_Spread_Function'])
        num_sf = num_hidden_units
        # CD Learning
        # Will get learned numpy arrays
        start = time.time()

        GBRBM = GaussianBernoulliRBM.GaussianBernoulliRBM(mini_batch, epoch, num_visible_units,
                                                          num_hidden_units,
                                                          sampling_type, sampling_times,
                                                          learning_rate,
                                                          momentum_rate, weight_decay_rate,
                                                          sparse_regularization, width_sf, height_sf, num_sf)

        C, B, W, sigma = GBRBM.contrastive_divergence_learning()

        # Measures time
        elapsed_time = time.time() - start
        h = elapsed_time // 3600
        m = (elapsed_time - h * 3600) // 60
        s = elapsed_time - h * 3600 - m * 60

        return C, B, W, sigma

    def save_result(self, C, B, W, sigma):
        """
        [1] Saves learning results
        [2] makes images of arrays
        :param C: biases of hidden units
        :param B: biases of visible units
        :param sigma: variance of gaussian
        :param W: weight
        :return: None
        """

        # [1] Saves learning results
        self.ex_file_handler.np_arr_save(os.path.join(self.dir_for_saving_result, 'C'), C)
        self.ex_file_handler.np_arr_save(os.path.join(self.dir_for_saving_result, 'B'), B)
        self.ex_file_handler.np_arr_save(os.path.join(self.dir_for_saving_result, 'sigma'), sigma)
        self.ex_file_handler.np_arr_save(os.path.join(self.dir_for_saving_result, 'W'), W)

        # [2] makes images of arrays
        for index, W in enumerate(W):
            path_to_file = os.path.join(self.dir_for_saving_result, 'W_%d.jpg' % index)

            Image.fromarray(
                np.uint8(np.reshape((W / np.max(W)) * 255, (self.width, self.height)))).save(
                path_to_file)
