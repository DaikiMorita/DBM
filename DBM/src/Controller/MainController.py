import configparser
from DBM.src.Manager import PreProcessManager
import shutil


class MainController(object):
    """
    Controls main process
    """

    def __init__(self):
        self.app_config = configparser.ConfigParser()
        self.app_config.read("application.ini")
        self.learning_config = configparser.ConfigParser()
        self.app_config.read("params.ini")

        # TO DO: is there any way automatically to create instances like java spring @Autowired?
        self.pre_process_manager = PreProcessManager.PreProcessManager()

    def start_main_pricess(self):
        """
        Main Process:
        :return:
        """

        ##################
        # 1. Preparation #
        ##################

        # self.pre_process_manager.make_dir()
        # shutil.copy2("application.ini", self.app_config["Setting"]["result_dir"])
