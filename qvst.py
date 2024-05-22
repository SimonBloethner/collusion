from mesa import Agent, Model
from mesa.time import BaseScheduler
from tqdm import tqdm
import numpy as np
from datetime import datetime


import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import matplotlib.cm as cm


# Hyper parameters
alpha = 0.3
gamma = 0.95
n_firms = 2
steps = 500 * 1000
runs = 20
multirun = True
k = 20
states = np.linspace(0, 1, k)
actions = np.linspace(0, 1, k)
increment = states[1] - states[0]
qs = np.zeros((states.shape[0], actions.shape[0]))

np.random.seed(66)
price_hist = np.zeros((steps, n_firms * (runs if multirun else 1)))
profit_hist = np.zeros((steps, n_firms * (runs if multirun else 1)))
demand_hist = np.zeros((steps, n_firms * (runs if multirun else 1)))


def demand(p_i, p_j):
    d = 1 - p_i + 0.5 * p_j
    return d


def profit(p_i, p_j):
    pi = p_i * demand(p_i, p_j)
    return pi


def tatonnement(delta_pi, delta_p, price, profit):
    if delta_pi > 0 and delta_p >= 0:
        price += increment
    if delta_pi > 0 and delta_p < 0:
        price -= increment
    if delta_pi < 0 and delta_p <= 0:
        price += increment
    if delta_pi < 0 and delta_p > 0:
        price -= increment
    if delta_pi == 0 and profit == 0:
        price = states[1]
    if price > 1:
        price = 1
    if price < 0:
        price = 0

    return price


class Firm(Agent):

    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.price = np.random.choice(actions, 1)
        self.profit = np.zeros((steps, 1))
        self.demand = np.zeros((steps, 1))
        self.price_list = np.zeros((steps, 1))
        self.price_list[0:2] = np.random.choice(actions, 2).reshape(2, 1)
        self.qmatrix = np.zeros((states.shape[0], actions.shape[0]))
        self.state = 0
        self.action = 0

    def act(self, p_j):
        if self.unique_id == 0:
            action = tatonnement(self.profit[model.period - 1] - self.profit[model.period - 2], self.price_list[model.period - 1] - self.price_list[model.period - 2], self.price, self.profit[model.period - 1])
        else:
            self.state = np.searchsorted(states, p_j)

            if model.epsilon > np.random.uniform(0, 1):
                action = np.random.choice(actions, 1)
            else:
                action = actions[np.argmax(self.qmatrix[self.state, :])]

        self.price = action
        self.action = np.searchsorted(actions, self.price)
        self.price_list[model.period:(model.period + 2)] = action

    def update(self, p_i, p_j, p_j_1):
        self.demand[model.period - 2] = demand(p_i, p_j)
        self.profit[model.period - 2] = profit(p_i, p_j)

        state = np.searchsorted(states, p_j)
        next_state = np.searchsorted(states, p_j_1)
        action = np.searchsorted(actions, p_i)

        new_est = self.profit[model.period - 2] + gamma * profit(p_i, p_j_1) + gamma ** 2 * np.max(self.qmatrix[next_state, :])
        new_val = (1 - alpha) * self.qmatrix[state, action] + alpha * new_est
        return new_val

    def observe(self, p_i, p_j, p_j_1):
        action = np.searchsorted(actions, p_i)
        state = np.searchsorted(states, p_j)

        # Is the - 2 in the indexing correct here?
        self.qmatrix[state, action] = self.update(p_i, p_j, p_j_1)

    def update_t(self, p_i, p_j):
        self.demand[model.period - 2:model.period] = demand(p_i, p_j)
        self.profit[model.period - 2:model.period] = profit(p_i, p_j)


class CollusionModel(Model):

    def __init__(self, N):
        self.num_agents = N
        self.period = 2
        # self.max_demand = max_demand
        self.demand_list = []
        self.epsilon = 1
        self.theta = 1 - np.power(0.001, 1 / (0.5 * steps))
        # Create agents
        self.schedule = BaseScheduler(self)
        for i in range(self.num_agents):
            a = Firm(i, self)
            self.schedule.add(a)
        for a in self.schedule.agents:
            a.demand[0:2] = demand(a.price_list[0:2], self.schedule.agents[0 ** a.unique_id].price_list[0:2])
            a.profit[0:2] = a.demand[0:2] * a.price_list[0:2]

    def step(self):
        a = model.schedule.agents[model.period % 2]
        if a.unique_id == 0:
            a.update_t(a.price_list[self.period - 2:self.period], self.schedule.agents[0 ** a.unique_id].price_list[self.period - 2:self.period])
        else:
            a.observe(a.price_list[self.period - 2], self.schedule.agents[0 ** a.unique_id].price_list[self.period - 2], self.schedule.agents[0 ** a.unique_id].price_list[self.period - 1])

        a.act(self.schedule.agents[0 ** a.unique_id].price_list[self.period])

        model.schedule.agents[0 ** a.unique_id].demand[model.period - 2] = demand(self.schedule.agents[0 ** a.unique_id].price_list[self.period - 2], a.price_list[self.period - 2])
        model.schedule.agents[0 ** a.unique_id].profit[model.period - 2] = profit(self.schedule.agents[0 ** a.unique_id].price_list[self.period - 2], a.price_list[self.period - 2])

        self.epsilon = (1 - self.theta) ** self.period
        self.period += 1
        if self.period == steps:
            agent_id = 0
            for a in self.schedule.agents:
                for time in range(steps-2, steps):
                    a.demand[time] = demand(a.price_list[time], self.schedule.agents[0 ** a.unique_id].price_list[time])
                    a.profit[time] = profit(a.price_list[time], self.schedule.agents[0 ** a.unique_id].price_list[time])
                price_hist[:, agent_id + j * n_firms] = a.price_list.reshape(-1)
                demand_hist[:, agent_id + j * n_firms] = a.demand.reshape(-1)
                profit_hist[:, agent_id + j * n_firms] = a.profit.reshape(-1)
                if a.unique_id == 1:
                    global qs
                    qs = qs + a.qmatrix
                agent_id += 1


model = CollusionModel(n_firms)

start = datetime.now()

if multirun:
    for j in range(runs):
        model.__init__(n_firms)
        for i in range(2, steps):
            model.step()

    mean_price = np.transpose(np.vstack([np.mean(price_hist[:, np.arange(0, price_hist.shape[1], 2)], axis=1), np.mean(price_hist[:, np.arange(1, price_hist.shape[1], 2)], axis=1)]))
    mean_demand = np.transpose(np.vstack([np.mean(demand_hist[:, np.arange(0, demand_hist.shape[1], 2)], axis=1), np.mean(demand_hist[:, np.arange(1, demand_hist.shape[1], 2)], axis=1)]))
    mean_profit = np.transpose(np.vstack([np.mean(profit_hist[:, np.arange(0, profit_hist.shape[1], 2)], axis=1), np.mean(profit_hist[:, np.arange(1, profit_hist.shape[1], 2)], axis=1)]))

    labels = ['Firm 1', 'Firm 2']
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True)
    ax1.plot(mean_price, label=['Firm 1', 'Firm 2'])
    ax1.title.set_text('Price')
    ax2.plot(mean_demand, label=['Firm 1', 'Firm 2'])
    ax2.title.set_text('Demand')
    ax3.plot(mean_profit, label=['Firm 1', 'Firm 2'])
    ax3.title.set_text('Profit')
    fig.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.4, hspace=0.4)
    fig.legend(labels, loc='upper right', bbox_transform=fig.transFigure)

    plt.savefig('collusion/results.png')

    min_val = 0
    max_val = 0
    min_val = np.min(qs) if np.min(qs) < min_val else min_val
    max_val = np.max(qs) if np.max(qs) > max_val else max_val

    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(8, 8))

    cmap = cm.get_cmap('viridis')
    normalizer = Normalize(min_val, max_val)
    im = cm.ScalarMappable(norm=normalizer)

    axes.set_xticks(range(actions.shape[0]))
    axes.set_yticks(range(actions.shape[0]))
    axes.set_xticklabels(np.round(actions, 2))
    axes.set_yticklabels(np.round(actions, 2))
    axes.imshow(qs, cmap=cmap, norm=normalizer)

    plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.4, hspace=0.6)

    fig.colorbar(im)
    plt.savefig('collusion/qmatrices.png')

    np.save('collusion/price.npy', price_hist)
    np.save('collusion/demand.npy', demand_hist)
    np.save('collusion/profit.npy', profit_hist)

    end = datetime.now()
    print('Start time was: {}'.format(start))
    print('End time was: {}'.format(end))
    print('Execution time was: {}'.format(datetime.now() - start))


if not multirun:
    for i in tqdm(range(2, steps)):
        model.step()

    labels = ['Firm 1', 'Firm 2']
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True)
    ax1.plot(price_hist[-1000:, :], label=['Firm 1', 'Firm 2'])
    ax1.title.set_text('Price')
    ax2.plot(demand_hist[-1000:, :], label=['Firm 1', 'Firm 2'])
    ax2.title.set_text('Demand')
    ax3.plot(profit_hist[-1000:, :], label=['Firm 1', 'Firm 2'])
    ax3.title.set_text('Profit')
    fig.legend(labels, loc='upper right', bbox_transform=fig.transFigure)
    fig.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.4, hspace=0.4)


    # agent_view = plt.figure(figsize=plt.figaspect(0.5))
    # for agent in range(n_firms):
    #     agent_plt = agent_view.add_subplot(int(np.ceil(n_firms / 3)), 3, agent + 1)
    #
    #     agent_plt.plot(profit_hist[-1000:, agent]), agent_plt.plot(price_hist[-1000:, agent]), agent_plt.plot(demand_hist[-1000:, agent])
    #     agent_plt.set_title('{}'.format(agent))
    #
    #
    # plt.legend(['Profit', 'Prices', 'Demand'], loc='lower right', bbox_to_anchor=(4, 0.0001))
    # plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.4, hspace=0.6)

    min_val = 0
    max_val = 0
    for a in model.schedule.agents:
        min_val = np.min(a.qmatrix) if np.min(a.qmatrix) < min_val else min_val
        max_val = np.max(a.qmatrix) if np.max(a.qmatrix) > max_val else max_val

    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(8, 8))

    cmap = cm.get_cmap('viridis')
    normalizer = Normalize(min_val, max_val)
    im = cm.ScalarMappable(norm=normalizer)

    for i, ax in enumerate(axes.flat):
        agent = model.schedule.agents[i]
        ax.imshow(agent.qmatrix, cmap=cmap, norm=normalizer)
        ax.set_title(str(i))
        ax.set_xticks(range(actions.shape[0]))
        ax.set_yticks(range(actions.shape[0]))
        ax.set_xticklabels(np.round(actions, 2))
        ax.set_yticklabels(np.round(actions, 2))
        if i + 1 == n_firms:
            break

    plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.4, hspace=0.6)

    fig.colorbar(im, ax=axes.ravel().tolist())
    plt.show()

