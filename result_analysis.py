import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import matplotlib.cm as cm


def demand(p_i, p_j):
    d = 1 - p_i + 0.5 * p_j
    return d


def profit(p_i, p_j):
    pi = p_i * demand(p_i, p_j)
    return pi


prices = np.load('/Users/Simon/price.npy')

demands = [demand(prices[:, i], prices[:, i + 1]) for i in range(0, prices.shape[1] - 1, 2)]
demands = np.transpose(np.array(demands))
profits = demands * prices[:, range(0, prices.shape[1], 2)]
profits = [profit(prices[:, i], prices[:, i + 1]) for i in range(prices.shape[1] - 1)]
