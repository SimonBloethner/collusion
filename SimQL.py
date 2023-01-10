from mesa import Agent, Model
from mesa.time import BaseScheduler
from tqdm import tqdm
import numpy as np
from datetime import datetime

from EconFunctions import demand

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import matplotlib.cm as cm


# Hyper parameters
alpha = 0.3
gamma = 0.95
n_firms = 2
steps = 500 * 1000
runs = 100
multirun = True
continuous = True
states = np.linspace(0, 1, 6)
actions = np.linspace(0, 1, 6)


np.random.seed(66)
price_hist = np.zeros((steps, n_firms * (runs if multirun else 1)))
profit_hist = np.zeros((steps, n_firms * (runs if multirun else 1)))
demand_hist = np.zeros((steps, n_firms * (runs if multirun else 1)))


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

    def act(self, rest):
        self.state = np.searchsorted(states, rest)

        if model.epsilon > np.random.uniform(0, 1):
            action = np.random.choice(actions, 1)
        else:
            action = actions[np.argmax(self.qmatrix[self.state, :])]

        self.price = action
        self.action = np.searchsorted(actions, self.price)
        self.price_list[model.period] = action
        model.prices[model.period, self.unique_id] = action

    def update(self, p_i, rest, rest_1):
        action = np.searchsorted(actions, p_i)
        state = np.searchsorted(states, rest)
        next_state = np.searchsorted(states, rest_1)

        pot_profit = p_i * demand(p_i, rest_1, n_firms)
        new_est = self.profit[model.period - 2] + gamma * pot_profit + gamma ** 2 * np.max(self.qmatrix[next_state, :])
        new_val = (1 - alpha) * self.qmatrix[state, action] + alpha * new_est
        return new_val

    def observe(self, p_i, rest, rest_1):
        action = np.searchsorted(actions, p_i)
        state = np.searchsorted(states, rest)

        self.qmatrix[state, action] = self.update(p_i, rest, rest_1)


class CollusionModel(Model):

    def __init__(self, N):
        self.num_agents = N
        self.period = 2
        # self.max_demand = max_demand
        self.demand_list = []
        self.prices = np.zeros([steps, n_firms])
        self.epsilon = 1
        self.theta = 1 - np.power(0.001, 1 / (0.5 * steps))
        # Create agents
        self.schedule = BaseScheduler(self)
        for i in range(self.num_agents):
            a = Firm(i, self)
            self.schedule.add(a)
        for a in self.schedule.agents:
            self.prices[0:2, a.unique_id] = a.price_list[0:2].reshape(2,)
        for a in self.schedule.agents:
            rest = np.mean(np.delete(self.prices[0:2, :], a.unique_id, axis=1), axis=1)   # Here is where we lose information due to rounding the mean.
            a.demand[0:2] = demand(a.price_list[0:2].reshape(2,), rest, n_firms).reshape(2, 1)
            a.profit[0:2] = a.demand[0:2] * a.price_list[0:2]

    def step(self):

        for a in self.schedule.agents:
            rest = np.mean(np.delete(self.prices[self.period - 2, :], a.unique_id))   # Here is where we lose information due to rounding the mean.
            rest_1 = np.mean(np.delete(self.prices[self.period - 1, :], a.unique_id))  # Here is where we lose information due to rounding the mean.

            a.observe(a.price_list[self.period - 2], rest, rest_1)
            a.act(np.mean(np.delete(self.prices[self.period, :], a.unique_id)))  # Here is where we lose information due to rounding the mean.

        for a in self.schedule.agents:
            rest = np.mean(np.delete(self.prices[self.period, :], a.unique_id))
            a.demand[self.period] = demand(a.price_list[self.period], rest, n_firms)
            a.profit[self.period] = a.demand[self.period] * a.price_list[self.period]

        self.epsilon = (1 - self.theta) ** self.period
        self.period += 1
        if self.period == steps:
            agent_id = 0
            for a in self.schedule.agents:
                for time in range(steps-2, steps):
                    rest = np.mean(np.delete(self.prices[time, :], a.unique_id))  # Here is where we lose information due to rounding the mean.
                    a.demand[time] = demand(a.price_list[time], rest, n_firms)
                    a.profit[time] = a.demand[time] * a.price_list[time]

                price_hist[:, agent_id + j * n_firms] = a.price_list.reshape(-1)
                demand_hist[:, agent_id + j * n_firms] = a.demand.reshape(-1)
                profit_hist[:, agent_id + j * n_firms] = a.profit.reshape(-1)
                agent_id += 1


model = CollusionModel(n_firms)

start = datetime.now()

if multirun:
    for j in range(runs):
        model.__init__(n_firms)
        for i in range(2, steps):
            model.step()

    mean_price = []
    for agent in range(n_firms):
        mean_price.append(np.mean(price_hist[:, np.arange(agent, price_hist.shape[1], n_firms)], axis=1))
    mean_price = np.transpose(np.asarray(mean_price))

    mean_demand = []
    for agent in range(n_firms):
        mean_demand.append(np.mean(demand_hist[:, np.arange(agent, demand_hist.shape[1], n_firms)], axis=1))
    mean_demand = np.transpose(np.asarray(mean_demand))

    mean_profit = []
    for agent in range(n_firms):
        mean_profit.append(np.mean(profit_hist[:, np.arange(agent, profit_hist.shape[1], n_firms)], axis=1))
    mean_profit = np.transpose(np.asarray(mean_profit))

    labels = ['Firm {}'.format(i) for i in np.arange(n_firms) + 1]

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True)
    ax1.plot(mean_price, label=labels)
    ax1.title.set_text('Price')
    ax2.plot(mean_demand, label=labels)
    ax2.title.set_text('Demand')
    ax3.plot(mean_profit, label=labels)
    ax3.title.set_text('Profit')
    fig.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.4, hspace=0.4)
    fig.legend(labels, loc='upper right', bbox_transform=fig.transFigure)

    plt.savefig('collusion/results_{}_{}.png'.format(n_firms, runs))

    np.save('collusion/price.npy', price_hist)
    np.save('collusion/demand.npy', demand_hist)
    np.save('collusion/profit.npy', profit_hist)

    end = datetime.now()
    print('Start time was: {}'.format(start))
    print('End time was: {}'.format(end))
    print('Execution time was {}'.format(datetime.now() - start))


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

