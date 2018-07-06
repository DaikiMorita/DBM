# coding=utf-8
import tqdm
import numpy as np
import warnings
import math


class GaussianBernoulliRBM(object):
    """
    Gaussian Bernoulli Restricted Boltzmann Machine
    Visible units and hidden units are modeled with Gaussian and Bernoulli distribution, respectively.
    Therefore, this machine can be applied into real values data without transforming the data into binary data.
    """

    # TO DO: is there any way automatically to create constructor like java-lombok?
    def __init__(self, num_v_unit, num_h_unit):
        """

        :param num_v_unit:
        :param num_h_unit:
        """
        self.num_v_unit = num_v_unit
        self.num_h_unit = num_h_unit

        # Initialization
        # W: weight (dim num hidden * num visible)
        # B: biases of visible units(dim 1 * num visible)
        # C: biases of hidden units(dim 1 * num hidden)
        # sigma: scalar or numpy array (dim 1 * visible units)

        self.W = np.ones((self.num_h_unit, self.num_v_unit))
        self.B = np.ones((1, self.num_v_unit))
        self.C = np.ones((1, self.num_h_unit))
        self.Sigma = np.ones((1, self.num_v_unit))

    def learning(self, train_data, max_epoch, sampling_times=1, mini_batch_size=10, sampling_type="CD",
                 learning_rate=0.01, momentum_rate=0, weight_decay_rate=0, sparse_regularization_target=0,
                 sparse_regularization_rate=0):
        """

        :param train_data:
        :param max_epoch:
        :param sampling_times:
        :param mini_batch_size:
        :param sampling_type:
        :param learning_rate:
        :param momentum_rate:
        :param weight_decay_rate:
        :param sparse_regularization_target:
        :param sparse_regularization_rate:
        :return:
        """
        W = self.W
        B = self.B
        C = self.C
        Sigma = self.Sigma

        # For momentum
        delta_W = 0
        delta_B = 0
        delta_C = 0

        rho_new = 0

        mini_batch = make_mini_batch(train_data, mini_batch_size)

        learning_params = {"sampling_type": sampling_type,
                           "sampling_times": sampling_times,
                           "learning_rate": learning_rate,
                           "momentum_rate": momentum_rate,
                           "weight_decay_rate": weight_decay_rate,
                           "sparse_regularization_target": sparse_regularization_target,
                           "sparse_reglarization_rate": sparse_regularization_rate}

        # initialization of X_k
        X_k = np.array(mini_batch[0])

        # CD Learning
        for e in tqdm.tqdm(range(0, max_epoch)):
            #####################
            # 1. Gibbs Sampling #
            #####################
            X = np.array(mini_batch[int(e % len(mini_batch))])

            X_k = gibbs_sampling(X, X_k, C, B, W, Sigma, learning_params["sampling_type"])

            ######################
            # 2. Gradient Update #
            ######################

            X, X_k, W, B, C, Sigma, delta_C, delta_B, delta_W, rho_new = gradient_update(X, X_k, W, B, C, Sigma,
                                                                                         delta_C, delta_B, delta_W,
                                                                                         rho_new, learning_params)
            # non negative limitation
            W[W < 0] = 0

        self.W = W
        self.B = B
        self.C = C
        self.Sigma = Sigma

    def get_params(self):
        """

        :return:
        """
        return self.W, self.B, self.C, self.Sigma


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

    data_array_length = len(data_list)
    rest = data_array_length % mini_batch_size

    mini_batches = [data_list[i:i + mini_batch_size] for i in
                    range(0, data_array_length - rest, mini_batch_size)]

    rest_batch = data_list[data_array_length - rest:] + data_list[0:mini_batch_size - rest]
    mini_batches.append(rest_batch)

    return mini_batches


def gradient_update(X, X_k, W_new, B_new, C_new, Sigma_new, delta_C, delta_B, delta_W, rho_new, learning_params):
    learning_rate = learning_params["learning_rate"]
    sparse_regularization_rate = learning_params["sparse_regularization_rate "]
    momentum_rate = learning_params["momentum_rate"]
    weight_decay_rate = learning_params["weight_decay_rate "]
    P_H_1_X = prob_H_1_X(X, C_new, W_new, Sigma_new)
    P_H_1_X_k = prob_H_1_X(X_k, C_new, W_new, Sigma_new)

    rho_old = rho_new
    C_old = C_new
    B_old = B_new
    # W_old = W_new * self.spread_funcs
    W_old = W_new
    sigma_old = Sigma_new

    rho_new, grad_E_sparse_W, grad_E_sparse_C = sparse_regularization(X,
                                                                      C_old, W_old,
                                                                      sigma_old, rho_old)

    C_new = C_old + learning_rate * (CD_C(P_H_1_X, P_H_1_X_k)
                                     - sparse_regularization_rate * grad_E_sparse_C) \
            + momentum_rate * delta_C

    B_new = B_old + learning_rate * CD_B(X, X_k, sigma_old) \
            + momentum_rate * delta_B

    W_new = W_old + learning_rate * (CD_W(X, X_k, P_H_1_X, P_H_1_X_k,
                                          sigma_old) - weight_decay_rate * W_old - sparse_regularization_rate * grad_E_sparse_W) \
            + momentum_rate * delta_W

    sigma_new = sigma_old

    delta_C = C_new - C_old
    delta_B = B_new - B_old
    delta_W = W_new - W_old

    return X, X_k, W_new, B_new, C_new, sigma_new, delta_C, delta_B, delta_W, rho_new


def gibbs_sampling(X, X_k, C_new, B_new, W_new, sigma_new, sampling_type):
    """

    :param X:
    :param X_k:
    :param C_new:
    :param B_new:
    :param W_new:
    :param sigma_new:
    :return:
    """

    if sampling_type == 'CD':
        X_k = block_gibbs_sampling(X, C_new, B_new, W_new, sigma_new)

    elif sampling_type == 'PCD':
        X_k = block_gibbs_sampling(X_k, C_new, B_new, W_new, sigma_new)

    return X_k


def sparse_regularization(X, C, W, sigma, rho_old, sparse_regularization_target):
    """

    :param sparse_regularization:
    :param X:
    :param C:
    :param W:
    :param sigma:
    :param P_H_1_X:
    :param H_X:
    :param sparse_regularization_target:
    :param rho_old:
    :return:
    """

    N = X.shape[0]

    # dim: 1 * num_hidden_units
    rho_new = 0.9 * rho_old + 0.1 * np.sum(prob_H_1_X(X, C, W, sigma), axis=0) / N

    delta_E_sparse_C = (-sparse_regularization_target / rho_new + (1 - sparse_regularization_target) / (
            1 - rho_new)) / N
    delta_E_sparse_C = delta_E_sparse_C[np.newaxis, :]

    S = np.empty((X.shape[0], delta_E_sparse_C.shape[1], X.shape[1]))
    for index, x in enumerate(X):
        S[index, :, :] = np.dot(delta_E_sparse_C.T, x[np.newaxis, :])

    delta_E_sparse_W = np.sum(S, axis=0) / N

    return rho_new, delta_E_sparse_W, delta_E_sparse_C


def block_gibbs_sampling(X, C, B, W, sigma, sampling_times):
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


def prob_H_1_X(X, C, W, sigma):
    """
    A row is a vector where i-th is the probability of h_i becoming 1 when given X
    :param X: values of visible (dim: num data * num visible units)
    :param C: biases of hidden units(dim 1 * num hidden)
    :param W: weight (dim num hidden * num visible)
    :param sigma: scalar or numpy array (dim 1 * visible units)
    :return: numpy array (dim: num data * num hidden)
    """

    warnings.filterwarnings('error')
    try:

        return 1 / (1 + np.exp(-C - (np.dot(X, np.transpose(W))) / (sigma * sigma)))

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


def sampling_X_H(H, B, W, sigma):
    """
    Gets samples of X following Gaussian distribution when given H
    :param H: values of hidden (dim: num data * num hidden)
    :param B: biases of visible (dim: num data * num visible)
    :param W: weight (dim num hidden * num visible)
    :param sigma: scalar or numpy array (dim 1 * visible units)
    :return: numpy array (dim: num data * num visible)
    """

    return sigma * np.random.randn(H.shape[0], W.shape[1]) + B + np.dot(H, W)


def CD_C(P_H_1_X, P_H_1_X_k):
    """
    Gradient approximation of C
    :param P_H_1_X: probability of H becoming 1 when given X
    :param P_H_1_X_k: probability of H becoming 1 when given X_k
    :return: numpy vector (dim: 1 * num_hidden_units)
    """

    return np.sum(P_H_1_X - P_H_1_X_k, axis=0) / P_H_1_X.shape[0]


def CD_B(X, X_k, Sigma):
    """
    Gradient approximation of B
    :param B: biases of visible (dim: num data * num visible)
    :param X_k: values of sampled visible (dim: num data * num visible units)
    :return: numpy vector (dim: 1 * num_visible_units)
    """

    return (np.sum(X - X_k, axis=0)) / (X.shape[0] * Sigma * Sigma)


def CD_W(X, X_k, P_H_1_X, P_H_1_X_k, Sigma):
    """
    Gradient approximation of W
    :param X: values of  visible (dim: num data * num visible units)
    :param X_k: values of sampled visible (dim: num data * num visible units)
    :param P_H_1_X: probability of H becoming 1 when given X
    :param P_H_1_X_k: probability of H becoming 1 when given X_k
    :return: numpy array(dim: num_hidden_units * num_visible_units)
    """

    # Numpy array was faster in some experiments.
    E = np.empty((X.shape[0], P_H_1_X.shape[1], X.shape[1]))

    for index, (P_x, x, P_x_k, x_k) in enumerate(zip((P_H_1_X),
                                                     (X),
                                                     (P_H_1_X_k),
                                                     (X_k))):
        E[index, :, :] = np.dot(P_x[:, np.newaxis], x[np.newaxis, :]) - np.dot(P_x_k[:, np.newaxis],
                                                                               x_k[np.newaxis, :])

    return np.sum(E, axis=0) / (X.shape[0] * Sigma * Sigma)


def CD_Sigma(X, X_k, P_H_1_X, P_H_1_X_k, B, W, Sigma):
    """
    Gradient approximation of sigma
    :param X: values of  visible (dim: num data * num visible units)
    :param X_k: values of sampled visible (dim: num data * num visible units)
    :param P_H_1_X: probability of H becoming 1 when given X
    :param P_H_1_X_k: probability of H becoming 1 when given X_k
    :param B: array (dim: num_data, num_visible_units)
    :param W: weight (dim num hidden * num visible)
    :param sigma: scalar or numpy array (dim 1 * visible units)
    :return: numpy array (dim: 1)
    """
    E_1_1 = np.sum(np.diag(np.dot((X - B), np.transpose((X - B)))), axis=0)
    E_1_2 = np.sum(np.diag(np.dot(X, np.transpose(W)) * P_H_1_X))

    E_2_1 = np.sum(np.diag(np.dot((X_k - B), np.transpose((X_k - B)))), axis=0)

    E_2_2 = np.sum(np.diag(np.dot(X_k, np.transpose(W)) * P_H_1_X_k))

    return (E_1_1 - 2 * E_1_2 - E_2_1 + 2 * E_2_2) / (X.shape[0] * Sigma * Sigma * Sigma)
