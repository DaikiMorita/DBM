from .Controller import ParamSetController


class Application(object):
    """
    Deep Boltzmann Machine Project
    """

    def __init__(self):
        ParamSetController.ParamSetController()


if __name__ == '__main__':
    Application = Application()
