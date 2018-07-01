class DeepBM(object):
    """
    Deep Boltzmann Machine
    """

    ##########
    # Notice #
    ##########

    # you can (or should) divide the methods below or add some additional methods
    # ,aiming at more simple and readable codes.

    # somewhere with less matrix calculation should be done by C/C++.

    def __init__(self, num_hidden_layer):
        self.num_hidden_layer = num_hidden_layer

    # To Do: this code block below will control the whole learning process described in the article.
    def learning(self):
        pass

    def pre_training(self):
        pass

    # To Do: must be better to add a method in charge of a calculation of KL divergence
    def variational_inference(self):
        pass

    def stochastic_approximation(self):
        pass

    def parameter_update(self):
        pass

    def get_W(self):
        pass

    def get_R(self):
        pass
