# coding=utf-8
import numpy as np


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

    def __init__(self, num_v_unit, num_h_layer, num_h_unit):
        self.num_v_unit = num_v_unit
        self.num_h_layer = num_h_layer
        self.num_h_unit = num_h_unit

        self.W = np.ones((self.num_h_layer, self.num_h_unit, self.num_v_unit))
        self.R = np.ones((self.num_h_layer, self.num_h_unit, self.num_v_unit))

    # To Do: this code block below will control the whole learning process described in the article.
    def learning(self, train_data, max_epoch, mini_batch_size=10, sampling_type="PCD", sampling_times=10,
                 learning_rate=0.01):
        # call pre_training

        # iterate T times below
        # variational_inference
        NU = calc_bottom_up_pass(R, V)
        MU = solve_mean_field_fixed_point_equation(W, V, mu, ite_K)

        # stochastic_approximation
        # parameter_update
        # decrease_alpha

    pass

    def get_model_params(self):
        """

        :return:
        """
        return self.W

    def set_model_params(self, W):
        """

        :param W:
        :return:
        """

        if W.shape == (self.num_h_layer, self.num_h_unit, self.num_v_unit):
            self.W = W
        else:
            raise ArrNotMatchException("raise")


def greedy_pre_training():
    pass


# we can run this block without "for" loop, using a matrix calculation
def variational_inference():
    pass


# To Do: eq. (8)~(10)
def sigmoid_activation(X):
    """

    :param X:
    :return:
    """
    return 1 / (1 + np.exp(-X))


# To Do: eq. (8)~(10)
def calc_bottom_up_pass(R, V):
    """

    :param R:
    :param V:
    :return:
    """
    num_h_layer = R.shape[0]

    # compensation for the lack of top-down feedback except the top layer
    comp = 2
    nu = V
    NU = []
    for n_h_l, r in enumerate(R):

        if n_h_l == num_h_layer:
            comp = 1

        nu = sigmoid_activation(np.sum(comp * r * nu, axis=1, keepdims=True))
        NU.append(nu)

    return NU


# To Do: eq. (4)~(6)
def solve_mean_field_fixed_point_equation(W, V, mu, ite_K):
    num_h_layer = W.shape[0]
    num_h_unit = W.shape[1]
    MU = []
    mu = V
    for k in list(range(ite_K)):
        for n_h_l in list(range(num_h_layer)):
            if n_h_l == num_h_layer:
                mu = sigmoid_activation(np.sum(W[n_h_l] * mu[n_h_l - 1], keepdims=True))
            else:
                mu = sigmoid_activation(np.sum(W[n_h_l] * mu, keepdims=True) + np.sum(W[n_h_l + 1] * mu[n_h_l + 1]))

            MU.append(mu)
    return MU


# To Do: eq. (11)
def calc_KL_divergence():
    pass


# we can run this block without "for" loop, using a matrix calculation
def stochastic_approximation():
    pass


# stochastic, hence may be better in python
def gibbs_sampler():
    pass


def parameter_update():
    pass


# To Do: inside () at parameter_update
def calc_gradient():
    pass


def decrease_alpha():
    pass


class ArrNotMatchException(Exception):
    def my_func(self):
        print("Array size does not match the vector size of h and v units")
