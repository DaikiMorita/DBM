class DeepBM(object):
    """
    Deep Boltzmann Machine
    """

    ##########
    # Notice #
    ##########

    # you can (or should) divide the methods below or add some additional methods,
    # aiming at more simple and readable codes and easier test codes.

    # somewhere with less matrix calculation or with too many "for" loops should be done by C/C++.
    # numpy is good at matrix calculation but runs too slowly inside "for" loop.

    def __init__(self, num_hidden_layer):
        self.num_hidden_layer = num_hidden_layer

    # To Do: this code block below will control the whole learning process described in the article.
    def learning(self):
        # call pre_training

        # iterate T times below
        # variational_inference
        # stochastic_approximation
        # parameter_update
        # decrease_alpha

        pass

    def greedy_pre_training(self):
        pass

    # we can run this block without "for" loop, using a matrix calculation
    def variational_inference(self):
        pass

    # To Do: eq. (8)~(10)
    def sigmoid_activation(self):
        pass

    # To Do: eq. (8)~(10)
    def calc_bottom_up_pass(self):
        pass

    # To Do: eq. (4)~(6)
    def solve_mean_field_fixed_point_equation(self):
        pass

    # To Do: eq. (11)
    def calc_KL_divergence(self):
        pass

    # we can run this block without "for" loop, using a matrix calculation
    def stochastic_approximation(self):
        pass

    # stochastic, hence may be better in python
    def gibbs_sampler(self):
        pass

    def parameter_update(self):
        pass

    # To Do: inside () at parameter_update
    def calc_gradient(self):
        pass

    def decrease_alpha(self):
        pass
