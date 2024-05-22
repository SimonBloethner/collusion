import numpy as np
from mesa import Model
from mesa.time import BaseScheduler
from sac import Agent as Firm


def demand(p_i, rest, num, demand_scale):
    d = (1 * demand_scale) - p_i + (1 / num) * rest
    return d


def logit_demand(p, a, a0, mu, demand_scale):
    e = np.exp((a - p) / mu)
    d = (e / (np.sum(e) + np.exp(a0 / mu))) * demand_scale
    return d


def profit(p_i, d):
    pi = p_i * d
    return pi


def global_econs(prices, rest, num, demand_scale, demand_type='linear', a=2, mu=0.25, a0=0):
    if demand_type == 'linear':
        d = 1 - prices + (1 / num) * rest
    elif demand_type == 'logit':
        e = np.exp((a - prices) / mu)
        d = (e / (np.sum(e) + np.exp(a0 / mu))) * demand_scale
    p = d * prices
    return d, p


class CollusionModelSimultaneous(Model):
    def __init__(self, n_firms, state_space, action_space, steps, run, load_checkpoint, demand_scale):
        self.state_space = state_space
        self.action_space = action_space
        self.n_firms = n_firms
        self.period = 2
        # self.max_demand = max_demand
        self.demand_list = []
        self.prices = np.zeros([steps, n_firms])
        self.steps = steps
        self.run = run
        self.load_checkpoint = load_checkpoint
        self.demand_scale = demand_scale

        # Create agents
        self.schedule = BaseScheduler(self)
        for i in range(self.n_firms):
            a = Firm(i, self, self.steps, self.state_space)
            self.schedule.add(a)
        for a in self.schedule.agents:
            self.prices[0:2, a.unique_id] = a.price_list[0:2].reshape(2, )
        for a in self.schedule.agents:
            rest = np.mean(np.delete(self.prices[0:2, :], a.unique_id, axis=1), axis=1)  # Here is where we lose information due to rounding the mean.
            a.demand[0:2] = demand(a.price_list[0:2].reshape(2, ), rest, n_firms, demand_scale=self.demand_scale).reshape(2, 1)
            a.profit[0:2] = a.demand[0:2] * a.price_list[0:2]

    def step(self):
        for a in self.schedule.agents:
            a.price = (a.choose_action(self.prices[self.period - 1, :]) * 0.5 + 0.5) * self.demand_scale
            a.price_list[self.period] = a.price
            self.prices[self.period, a.unique_id] = a.price

        for a in self.schedule.agents:
            rest = np.sum(np.delete(self.prices[self.period, :], a.unique_id))
            a.demand[self.period] = demand(a.price_list[self.period], rest, self.n_firms, demand_scale=self.demand_scale)
            a.profit[self.period] = a.demand[self.period] * a.price_list[self.period]

            observation = self.prices[self.period - 1, :]
            observation_ = self.prices[self.period, :]

        for a in self.schedule.agents:
            a.remember(observation, a.action, a.profit[self.period][0], observation_, 0 if self.period != self.steps else 1)
            if not self.load_checkpoint:
                a.learn()
        self.period += 1

    def transcribe(self, price_hist, demand_hist, profit_hist):

        agent_id = 0
        for a in self.schedule.agents:
            for time in range(self.steps - 2, self.steps):
                rest = np.sum(np.delete(self.prices[time, :], a.unique_id))
                a.demand[time] = demand(a.price_list[time], rest, self.n_firms, demand_scale=self.demand_scale)
                a.profit[time] = a.demand[time] * a.price_list[time]

            price_hist[:, agent_id + self.run * self.n_firms] = a.price_list.reshape(-1)
            demand_hist[:, agent_id + self.run * self.n_firms] = a.demand.reshape(-1)
            profit_hist[:, agent_id + self.run * self.n_firms] = a.profit.reshape(-1)
            agent_id += 1
        return price_hist, demand_hist, profit_hist
