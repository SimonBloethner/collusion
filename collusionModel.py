import numpy as np
from gym import spaces

np.random.seed(123)


def logit_demand(p, a, a0, mu, demand_scale):
    e = np.exp((a - p) / mu)
    d = (e / (np.sum(e) + np.exp(a0 / mu))) * demand_scale
    return d


def linear_demand(demand_scale, p_i, num, rest):
    d = (1 * demand_scale) - p_i + (1 / num) * rest
    return d


def market_outcome(p_i, rest, num, demand_scale, demand):
    if demand == 'linear':
        d = linear_demand(demand_scale, p_i, num, rest)
    elif demand == 'logit':
        d = logit_demand(p_i, a=2, a0=0, mu=0.25, demand_scale=1)
    p = d * p_i
    return d, p


class CollusionModelSimultaneous:
    def __init__(self, demand_scale, n_firms=2, lookback=1):
        self.state_space = n_firms * lookback
        self.action_bounds = (0, 1 * demand_scale)
        self.n_firms = n_firms
        self.period = 0
        self.demand_scale = demand_scale
        self.lookback = lookback
        self.action_space = spaces.Box(low=0, high=1*3, shape=(1,))
        self.name = 'collusion'

    def step(self, prices):

        rest = np.zeros(self.n_firms)
        for price in range(prices.shape[0]):
            rest[price] = np.sum(np.delete(prices, price))

        demands, profits = market_outcome(prices, rest, self.n_firms, self.demand_scale, demand='logit')

        self.period += 1

        return prices, demands, profits

    def initialize(self):
        prices = np.zeros((self.lookback, self.n_firms))
        demands = np.zeros((self.lookback, self.n_firms))
        profits = np.zeros((self.lookback, self.n_firms))

        for period in range(self.lookback):
            price = np.random.uniform(0, 1 * self.demand_scale, self.n_firms)
            prices[period, :], demands[period, :], profits[period, :] = self.step(price)

        return prices, demands, profits

    def sample_action(self):
        prices = np.zeros(self.n_firms)
        for firm in range(self.n_firms):
            prices[firm] = self.action_space.sample()
        prices, demands, profits = self.step(prices)

        return prices, demands, profits
