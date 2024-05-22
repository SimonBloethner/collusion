import numpy as np
from mesa import Agent, Model
from mesa.time import BaseScheduler
from tqdm import tqdm
from datetime import datetime
from itertools import product

from EconFunctions import global_econs

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import matplotlib.cm as cm


# Hyper parameters
alpha = 0.3
gamma = 0.95
n_firms = 2
steps = 1000 * 1000
runs = 2
multirun = False
continuous = True
k = 7
memory = 1
demand_scale = 10
actions = np.linspace(0, 1, k)
states = np.array(np.meshgrid(actions, actions)).T.reshape(-1, n_firms)
state_space = (k * memory) ** n_firms


np.random.seed(66)
price_hist = np.zeros((steps, n_firms, runs)) if multirun else np.zeros((steps, n_firms))
profit_hist = np.zeros((steps, n_firms, runs)) if multirun else np.zeros((steps, n_firms))
demand_hist = np.zeros((steps, n_firms, runs)) if multirun else np.zeros((steps, n_firms))


class Firm(Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.qmatrix = np.zeros((memory, actions.shape[0], actions.shape[0], actions.shape[0]))
        self.state = 0
        self.action = 0

    def act(self):
        if model.epsilon > np.random.uniform(0, 1):
            action = np.random.choice(actions, 1)[0]
        else:
            action = actions[np.argmax(self.qmatrix[(0,) + tuple(model.actions)])]

        self.action = np.array([np.searchsorted(actions, action)])
        model.next_actions[self.unique_id] = self.action
        model.prices[model.period, self.unique_id] = action

    def update(self, action):
        new_est = model.period_profit[self.unique_id] + gamma * np.max(self.qmatrix[(0,) + tuple(model.next_actions)])
        new_val = (1 - alpha) * self.qmatrix[(0,) + tuple(model.actions) + tuple(action)] + alpha * new_est
        return new_val

    def observe(self):
        self.qmatrix[(0,) + tuple(model.actions) + tuple(self.action)] = self.update(self.action)


class CollusionModel(Model):

    def __init__(self, N):
        self.num_agents = N
        self.period = 0
        # self.max_demand = max_demand
        self.prices = np.zeros([steps, n_firms])
        self.epsilon = 1
        self.dim = tuple([k for agent in range(self.num_agents)])
        self.theta = 1 - np.power(0.001, 1 / (0.5 * steps))
        self.actions = np.random.randint(0, k, 2)
        self.next_actions = np.zeros(self.num_agents).astype(int)
        self.pis = None
        self.demands = None
        self.agent_demands = np.zeros([steps, self.num_agents])
        self.agent_profits = np.zeros([steps, self.num_agents])
        self.period_profit = np.zeros(self.num_agents)

        self.demands, self.pis = self.inits()

        # Create agents
        self.schedule = BaseScheduler(self)
        for new_agent in range(self.num_agents):
            a = Firm(new_agent, self)
            self.schedule.add(a)

    def inits(self):
        PI = np.zeros([k, k, self.num_agents])
        demands = np.zeros([k, k, self.num_agents])
        for s in product(*[range(dimension) for dimension in self.dim]):
            prices = np.array([actions[action] for action in s])
            rest = np.array([np.sum(np.delete(prices, agent)) for agent in range(self.num_agents)])
            demands[s], PI[s] = global_econs(prices, rest, self.num_agents, demand_scale, demand_type='linear')
        return demands, PI

    def step(self):
        for a in self.schedule.agents:
            a.act()

        profits = self.pis[tuple(self.next_actions)]

        self.agent_demands[self.period, :] = self.demands[tuple(self.next_actions)]
        self.agent_profits[self.period, :] = profits

        self.period_profit = profits

        for a in self.schedule.agents:
            a.observe()

        self.actions = self.next_actions.copy()

        self.epsilon = np.exp(-0.00001 * self.period)   # (1 - self.theta) ** self.period
        self.period += 1



model = CollusionModel(n_firms)

start = datetime.now()

if multirun:
    for j in range(runs):
        model.__init__(n_firms)
        for i in range(steps):
            model.step()
        price_hist[:, :, j] = model.prices
        demand_hist[:, :, j] = model.agent_demands
        profit_hist[:, :, j] = model.agent_profits
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

    plt.savefig('collusion/figures/results_{}_{}.png'.format(n_firms, runs))

    np.save('collusion/outData/price.npy', price_hist)
    np.save('collusion/outData/demand.npy', demand_hist)
    np.save('collusion/outData/profit.npy', profit_hist)

    end = datetime.now()
    print('Start time was: {}'.format(start))
    print('End time was: {}'.format(end))
    print('Execution time was {}'.format(datetime.now() - start))


if not multirun:
    for i in tqdm(range(2, steps)):
        model.step()

    price_hist = model.prices
    demand_hist = model.agent_demands
    profit_hist = model.agent_profits

    labels = ['Firm 1', 'Firm 2']
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True)
    ax1.plot(price_hist, label=['Firm 1', 'Firm 2'])
    ax1.title.set_text('Price')
    ax2.plot(demand_hist, label=['Firm 1', 'Firm 2'])
    ax2.title.set_text('Demand')
    ax3.plot(profit_hist, label=['Firm 1', 'Firm 2'])
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

    # min_val = 0
    # max_val = 0
    # for a in model.schedule.agents:
    #     min_val = np.min(a.qmatrix) if np.min(a.qmatrix) < min_val else min_val
    #     max_val = np.max(a.qmatrix) if np.max(a.qmatrix) > max_val else max_val
    #
    # fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(8, 8))
    #
    # cmap = cm.get_cmap('viridis')
    # normalizer = Normalize(min_val, max_val)
    # im = cm.ScalarMappable(norm=normalizer)
    #
    # for i, ax in enumerate(axes.flat):
    #     agent = model.schedule.agents[i]
    #     ax.imshow(agent.qmatrix[0, :, :, :], cmap=cmap, norm=normalizer)
    #     ax.set_title(str(i))
    #     ax.set_xticks(range(actions.shape[0]))
    #     ax.set_yticks(range(actions.shape[0]))
    #     ax.set_xticklabels(np.round(actions, 2))
    #     ax.set_yticklabels(np.round(actions, 2))
    #     if i + 1 == n_firms:
    #         break
    #
    # plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.4, hspace=0.6)
    #
    # fig.colorbar(im, ax=axes.ravel().tolist())
    # plt.show()

