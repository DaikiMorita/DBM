import configparser
from datetime import datetime
from DBM.src.Manager import PreProcessManager
from DBM.src.Manager import ExFileManager
from DBM.src.Manager import DataProcessManager
import os


class MainController(object):
    """
    Controls main process
    """

    def __init__(self):

        self.app_conf = configparser.ConfigParser()
        self.app_conf.read("application.ini")

        self.learn_conf = configparser.ConfigParser()
        self.learn_conf.read("learning.ini")

        # TO DO: is there any way automatically to create instances like java spring @Autowired?
        self.data_process_manager = DataProcessManager.DataProcessManager()
        self.pre_process_manager = PreProcessManager.PreProcessManager()
        self.ex_file_manager = ExFileManager.ExFileManager()

    # To Do: conduct integration test as much as possible
    def start_main_pricess(self):
        """
        Main Process:
        :return:
        """

        ##################
        # 1. Preparation #
        ##################

        self.preparation()

        #################
        # 2. Reads data #
        #################

        # all_labels, all_data, each_label_data = self.read_data(self.app_conf["Setting"]["path_data_dir"])

        ##################
        # 3. Pre-process #
        ##################

        # data_train_batch, label_train_batch, data_train, data_test = self.pre_process_data(all_labels, all_data,
        #                                                                                    self.learn_conf["General"][
        #                                                                                        "ini_batch_size"])

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

    def preparation(self):
        """
        Preparation.
        [1] Makes a dir for saving results.
        [2] Copies config file
        :return: None

        """

        # [1] makes a dir for saving results
        self.data_process_manager.make_dir(self.app_conf["Setting"]["path_result_dir"],
                                           datetime.now().strftime("%Y-%m-%d_%H:%M:%S"))

        # [2] copy learning.ini into a result dir
        before_path = os.path.join(self.app_conf["Setting"]["path_learn_ini"],
                                   self.app_conf["Setting"]["name_learn_ini"])

        after_path = os.path.join(self.app_conf["Setting"]["path_result_dir"],
                                  datetime.now().strftime("%Y-%m-%d_%H:%M:%S"))
        self.data_process_manager.copy_file(before_path, after_path)

    def read_data(self, path_train_dirs):
        """
        Reads data.
        :return: num_all_data, all_data_array, each_label_data_array
        """

        # num_all_data: float scalar, total amount of data
        # all_data_array: 2-d list (dim: num_all_data * each data dimension)
        # each_label_data_array: 3-d list (dim: num_label*num_all_data * each data dimension)
        formatted_labels, formatted_data, each_label_data = self.ex_file_manager.read_image_data(path_train_dirs)

        return formatted_labels, formatted_data, each_label_data

    def pre_process_data(self, all_labels, all_data, mini_batch_size):
        """
        Pre-process od data
        [1] Normalizes data (changes it into data with mean 0 and variance 1)
        [2] Makes mini batches
        [3] Makes a dictionary with keys for learning
        :param all_data_array: 2-d list (dim: num all data * data dimension)
        :return: all_data, dict_data_parameter
        """

        # [1] Normalizes data
        all_data = self.pre_precess_manager.normalization(all_data)

        data_train, data_test, label_train, label_test = train_test_split(all_data, all_labels, test_size=0.1,
                                                                          shuffle=True)

        # [2] makes mini-batches where data with different labels will be contained at the almost same rate.
        data_train_batch = self.pre_precess_manager.make_mini_batch(data_train, mini_batch_size)
        label_train_batch = self.pre_precess_manager.make_mini_batch(label_train, mini_batch_size)

        return data_train_batch, label_train_batch, data_train, data_test

    def CD_learning(self, mini_batch):
        """
        Contastive divergence learning
        :param dict_data_parameter: dictionary with keys for learning
        :return: C, B, W, sigma
        """

        ini_file = configparser.ConfigParser()
        ini_file.read(self.config_file_name)

        ######################
        # General Parameters #
        ######################

        epoch = int(ini_file.get('GeneralParameter', 'Epoch'))

        num_visible_units = self.width * self.height
        num_hidden_units = int(ini_file.get('GeneralParameter', 'Num_Hidden_Unit'))
        learning_rate = float(ini_file.get('GeneralParameter', 'Learning_Rate'))

        ############
        # sampling #
        ############
        sampling_times = int(ini_file.get('SpecialParameter', 'Smapling_Times'))

        sampling_type = ''.join(ini_file['SpecialParameter']['Smapling_Type'])
        if not (sampling_type == 'CD' or sampling_type == 'PCD'):
            self.viewer.display_message("set CD or PCD to a param of sampling_type")
            raise Exception

        ############
        # momentum #
        ############
        if ''.join(ini_file['SpecialParameter']['Momentum']) == 'Yes':
            momentum_rate = float(ini_file.get('SpecialParameter', 'Momentum_Rate'))
        else:
            momentum_rate = 0

        ################
        # weight_decay #
        ################
        if ''.join(ini_file['SpecialParameter']['Weight_Decay']) == 'Yes':
            weight_decay_rate = float(ini_file.get('SpecialParameter', 'Weight_Decay_Rate'))
        else:
            weight_decay_rate = 0

        #########################
        # sparse_regularization #
        #########################
        if ''.join(ini_file['SpecialParameter']['Sparse_Regularization']) == 'Yes':

            sparse_regularization = (float(ini_file.get('SpecialParameter', 'Sparse_Regularization_Target')),
                                     float(ini_file.get('SpecialParameter', 'Sparse_Regularization_Rate')))

        else:
            sparse_regularization = (0, 0)

        width_sf = int(ini_file.get('SpecialParameter', 'Width_Spread_Function'))
        height_sf = int(ini_file.get('SpecialParameter', 'Height_Spread_Function'))
        num_sf = num_hidden_units
        # CD Learning
        # Will get learned numpy arrays
        start = time.time()

        GBRBM = GaussianBinaryRBM.GaussianBinaryRBM(mini_batch, epoch, num_visible_units,
                                                    num_hidden_units,
                                                    sampling_type, sampling_times,
                                                    learning_rate,
                                                    momentum_rate, weight_decay_rate,
                                                    sparse_regularization, width_sf, height_sf, num_sf,
                                                    self.dir_for_saving_result)

        self.viewer.display_message("\nContrastive Divergence Learning Starts...\n")

        C, B, W, sigma = GBRBM.Learning()

        # Measures time
        elapsed_time = time.time() - start
        h = elapsed_time // 3600
        m = (elapsed_time - h * 3600) // 60
        s = elapsed_time - h * 3600 - m * 60
        self.viewer.display_message(
            "\nContrastive Divergence Learning Finished...\n" + "About %d h %d m %d s \n" % (h, m, s))
        self.line_ui_manager.send_line("\nCD Learning \nAbout %d h %d m %d s" % (h, m, s))

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

        self.viewer.display_message("Results saving Starts\n")

        # [1] Saves learning results
        self.exfile_manager.numpy_array_save(os.path.join(self.dir_for_saving_result, 'C'), C)
        self.exfile_manager.numpy_array_save(os.path.join(self.dir_for_saving_result, 'B'), B)
        self.exfile_manager.numpy_array_save(os.path.join(self.dir_for_saving_result, 'sigma'), sigma)
        self.exfile_manager.numpy_array_save(os.path.join(self.dir_for_saving_result, 'W'), W)

        # [2] makes images of arrays
        for index, W in enumerate(W):
            path_to_file = os.path.join(self.dir_for_saving_result, 'W_%d.jpg' % index)

            Image.fromarray(
                np.uint8(np.reshape((W / np.max(W)) * 255, (self.width, self.height)))).save(
                path_to_file)

            self.line_ui_manager.send_line('W_%d.jpg' % index, path_to_file)

        self.viewer.display_message("Results saving Starts Finished \n")

    def test(self, C, W, sigma, label_list, H_list):
        """
        Test.
        Reads test data.
        Get H-s corresponding to the data.
        Campares the H-s to the learned H-s
        And estimates labels to the H-s.
        :param C: biases of hidden units
        :param W: weight
        :param sigma: variance of gaussian
        :param label_list: columns correspond to each H in the same index, respectively
        :param H_list: columns correspond to lables in the same index, respectively
        :return: None
        """

        for a_dir_test in os.listdir(self.path_test_dir):

            num_data, data_array = self.exfile_manager.get_data(os.path.join(self.path_test_dir, a_dir_test))
            estimated_labels = self.post_process_manager.estimate_data_category(C, W, sigma, label_list, H_list,
                                                                                data_array)
            for (index, data), estimated_label in zip(enumerate(data_array), estimated_labels):
                path_to_file = os.path.join(self.dir_for_saving_result, '%s_%d.jpg' % (a_dir_test, index))

                data = (data / np.max(data)) * 255

                Image.fromarray(np.uint8(
                    (np.reshape(data, (self.width, self.height))))).save(
                    path_to_file)
                self.line_ui_manager.send_line(
                    'RBM: %s\nTRUE: %s\n' % (estimated_label, a_dir_test,),
                    path_to_file)
