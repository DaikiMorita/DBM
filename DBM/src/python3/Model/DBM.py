# coding=utf-8
import numpy as np
import tqdm
from DBM.DBM.src.python3.Model import GBRBM


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
    def learning(self, train_data, max_epoch, max_epoch_rbm=50000, mini_batch_size=10, sampling_type="PCD",
                 sampling_times=10,
                 learning_rate=0.01):

        params_dbm = {}
        params_rbm = {}

        # call pre_training

        self.R = greedy_pre_training(self.num_v_unit, self.num_h_unit, train_data, max_epoch_rbm)

        # makes mini batch from train_data
        train_data = make_mini_batch(train_data, mini_batch_size)

        # ENTRY of Main Training
        for current_epoch in tqdm.tqdm(list(range(max_epoch))):
            # iterate T times below
            # variational_inference
            NU = calc_bottom_up_pass(R, V)
            MU = solve_mean_field_fixed_point_equation(W, V, mu, ite_K)

            # stochastic_approximation

            # parameter_update
            W = parameter_update()

            # decrease_learning_rate
            learning_rate = decrease_learning_rate(learning_rate, current_epoch, max_epoch, 0.1)
        # END of Main Training

        self.W = W

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


def make_mini_batch(data_list, mini_batch_size):
    """
    makes mini bathes from list-type data_array
    :param data_list: 2-d list
    :param mini_batch_size: size of mini batches
    :return:3d-list. A returned list will contain mini batches.
    Each batch will contain lists as you specified at the parameter.
    """

    # Now that data_array was shuffled,
    # mini batches will contain data with a different label at the almost same rate, statistically,
    # even when mini batches are made by extracting data from the top of the list 'data_array'

    data_list = data_list.tolist()
    data_array_length = len(data_list)
    rest = data_array_length % mini_batch_size

    mini_batches = [data_list[i:i + mini_batch_size] for i in
                    range(0, data_array_length - rest, mini_batch_size)]

    rest_batch = data_list[data_array_length - rest:] + data_list[0:mini_batch_size - rest]
    mini_batches.append(rest_batch)

    return np.array(mini_batches)


def greedy_pre_training(num_v_unit, num_h_unit, train_data, max_epoch_rbm):
    """

    :param num_v_unit:
    :param num_h_unit:
    :param train_data:
    :param max_epoch_rbm:
    :return:
    """
    pretrained_W = []
    V = num_v_unit
    for n in list(range(num_h_unit)):
        if n != 0:
            V = num_h_unit[n - 1]
        H = num_h_unit[n]
        rbm = GBRBM.GaussianBernoulliRBM(V, H)
        rbm.learning(train_data, max_epoch_rbm)
        pretrained_W.append(rbm.get_W())

    return pretrained_W


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
    gibbs_sampling()


def gibbs_sampling(X, X_k, W, B, C, Sigma, sampling_type, sampling_times):
    """

    :param X:
    :param X_k:
    :param W:
    :param B:
    :param C:
    :param Sigma:
    :param sampling_type:
    :param sampling_times:
    :return:
    """

    if sampling_type == 'PCD':
        X_k = block_gibbs_sampling(X, W, B, C, Sigma, sampling_times)

    elif sampling_type == 'CD':
        X_k = block_gibbs_sampling(X_k, W, B, C, Sigma, sampling_times)

    return X_k


def block_gibbs_sampling(X, W, B, C, sigma, sampling_times):
    """
    Block Gibbs Sampling
    :param X: values of visible (dim: num data * num visible units)
    :param C: biases of hidden units(dim 1 * num hidden)
    :param B: biases of visible units(dim 1 * num visible)
    :param W: weight (dim num hidden * num visible)
    :param sigma: scalar or numpy array (dim 1 * visible units)
    :return: sampled and averaged visible values X
    """

    temp = np.zeros((X.shape[0], X.shape[1]))
    X_k = X
    for _ in list(range(0, sampling_times)):
        H_k_1_X = prob_H_1_X(X_k, C, W, sigma)
        H_k = sampling_H_X(H_k_1_X)
        X_k = sampling_X_H(H_k, B, W, sigma)
        temp += X_k

    return temp / sampling_times


def prob_H_1_X(X, W, C, Sigma):
    """
    A row is a vector where i-th is the probability of h_i becoming 1 when given X
    :param X: values of visible (dim: num data * num visible units)
    :param W: weight (dim num hidden * num visible)
    :param C: biases of hidden units(dim 1 * num hidden)
    :param Sigma: scalar or numpy array (dim 1 * visible units)
    :return: numpy array (dim: num data * num hidden)
    """

    warnings.filterwarnings('error')
    try:

        return 1 / (1 + np.exp(-C - (np.dot(X, np.transpose(W))) / (Sigma * Sigma)))

    except RuntimeWarning as warn:

        # Over float is interpreted as RuntimeWarning.
        # An array filled with 0 will be returned instead of the array with over floated number.
        return np.zeros((X.shape[0], W.shape[0]))


def sampling_H_X(P_H_1):
    """
    Gets samples of H following Bernoulli distribution when given X
    :param P_H_1: probability of H becoming 1 when given X
    :return: array (dim: num_data*num_hidden_units)
    """

    return np.fmax(np.sign(P_H_1 - np.random.rand(P_H_1.shape[0], P_H_1.shape[1])),
                   np.zeros((P_H_1.shape[0], P_H_1.shape[1])))


def sampling_X_H(H, W, B, Sigma):
    """
    Gets samples of X following Gaussian distribution when given H
    :param H: values of hidden (dim: num data * num hidden)
    :param W: weight (dim num hidden * num visible)
    :param B: biases of visible (dim: num data * num visible)
    :param Sigma: scalar or numpy array (dim 1 * visible units)
    :return: numpy array (dim: num data * num visible)
    """

    return Sigma * np.random.randn(H.shape[0], W.shape[1]) + B + np.dot(H, W)


def parameter_update():
    pass


# To Do: inside () at parameter_update
def calc_gradient():
    pass


def decrease_learning_rate(learning_rate, current_epoch, max_epoch, target):
    """

    :param learning_rate:
    :param current_epoch:
    :param max_epoch:
    :param target:
    :return:
    """
    rate = (1 - target) / (max_epoch * max_epoch) * (current_epoch - max_epoch) * (current_epoch - max_epoch) + target
    return learning_rate * rate


class ArrNotMatchException(Exception):
    def my_func(self):
        print("Array size does not match the vector size of h and v units")
