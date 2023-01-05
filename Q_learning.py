import random

from mesa import Agent, Model
from mesa.time import RandomActivation
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import matplotlib.cm as cm

np.random.seed(2)
n_firms = 10
steps = 10**7
max_demand = 100

states = np.linspace(0, max_demand, 10)
actions = np.linspace(0, max_demand, 10)

# Hyperparameters
alpha = 0.3
gamma = 0.998
epsilon = 0.3

price_hist = np.zeros((steps, n_firms))
profit_hist = np.zeros((steps, n_firms))
demand_hist = np.zeros((steps, n_firms))


def q_function(x, alpha, gamma, profit, next_max):
    new_val = (1 - alpha) * x + alpha * (profit + gamma * next_max)
    return new_val


class Firm(Agent):

    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.price = np.random.uniform(1, 50, 1)
        self.profit = np.zeros((steps, 1))
        self.demand = np.zeros((steps, 1))
        self.price_list = np.zeros((steps, 1))
        self.qmatrix = np.zeros((states.shape[0], actions.shape[0]))
        self.state = 0

    def act(self):
        self.state = np.searchsorted(states, model.avg_price)
        if model.period == 0:
            action = np.random.choice(actions, 1)
        else:
            if model.epsilon > np.random.uniform(0, 1):
                action = np.random.choice(actions, 1)
            else:
                action = actions[np.argmax(self.qmatrix[self.state, :])]

        self.price = action
        self.price_list[model.period] = action
        model.price_list[model.period, self.unique_id] = self.price

    def observe(self):
        next_state = np.searchsorted(states, model.avg_price)
        action = np.searchsorted(actions, self.price)

        self.profit[model.period] = self.demand[model.period] * self.price
        self.qmatrix[self.state, action] = q_function(self.qmatrix[self.state, action], alpha, gamma, self.profit[model.period], np.max(self.qmatrix[next_state, :]))


class CollusionModel(Model):

    def __init__(self, N, mode):
        self.num_agents = N
        self.mode = mode
        self.period = 0
        self.max_demand = max_demand
        self.demand_list = []
        self.price_list = np.vstack([np.random.uniform(1, 50, n_firms).reshape(1, n_firms), np.zeros((steps - 1, n_firms))])
        self.avg_price = 0
        self.epsilon = epsilon
        # Create agents
        self.schedule = RandomActivation(self)
        for i in range(self.num_agents):
            a = Firm(i, self)
            self.schedule.add(a)

    def step(self):
        for a in self.schedule.agents:
            a.act()

        self.quantity_allocation()

        for a in self.schedule.agents:
            a.demand[self.period] = self.demand_list[a.unique_id]
            a.observe()
        self.schedule.step()

        self.period += 1
        if self.period % (steps * 0.1) == 0 and self.period != 0:
            self.epsilon *= 0.9
        if self.period == steps:
            agent_id = 0
            for a in self.schedule.agents:
                profit_hist[:, agent_id] = a.profit.reshape(-1)
                price_hist[:, agent_id] = a.price_list.reshape(-1)
                demand_hist[:, agent_id] = a.demand.reshape(-1)
                agent_id += 1

    def quantity_allocation(self):
        self.demand_list = np.array([self.max_demand - self.price_list[self.period, i] + (0.1 / 9) * (np.sum(self.price_list[self.period, :]) - self.price_list[self.period, i]) for i in range(n_firms)])
        self.demand_list[self.demand_list < 0] = 0
        self.avg_price = np.mean(self.price_list[self.period, :])


model = CollusionModel(n_firms, mode='proportional')

for i in tqdm(range(steps)):
    model.step()

fig = plt.figure(figsize=plt.figaspect(0.5))
profit_plt = fig.add_subplot(3, 1, 1)
profit_plt.plot(profit_hist[-1000:, :])
profit_plt.set_title('Profits')

price_plt = fig.add_subplot(3, 1, 2)
price_plt.plot(price_hist[-1000:, :])
price_plt.set_title('Prices')

demand_plt = fig.add_subplot(3, 1, 3)
demand_plt.plot(demand_hist[-1000:, :])
demand_plt.set_title('Demand')

plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.4, hspace=0.6)

agent_view = plt.figure(figsize=plt.figaspect(0.5))
for agent in range(n_firms):
    agent_plt = agent_view.add_subplot(int(np.ceil(n_firms / 3)), 3, agent + 1)

    agent_plt.plot(profit_hist[-1000:, agent]), agent_plt.plot(price_hist[-1000:, agent]), agent_plt.plot(demand_hist[-1000:, agent])
    agent_plt.set_title('{}'.format(agent))


plt.legend(['Profit', 'Prices', 'Demand'], loc='lower right', bbox_to_anchor=(4, 0.0001))
plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.4, hspace=0.6)

min_val = 0
max_val = 0
for a in model.schedule.agents:
    min_val = np.min(a.qmatrix) if np.min(a.qmatrix) < min_val else min_val
    max_val = np.max(a.qmatrix) if np.max(a.qmatrix) > max_val else max_val

fig, axes = plt.subplots(nrows=int(np.ceil(n_firms / 3)), ncols=3, figsize=(8, 8))

cmap = cm.get_cmap('viridis')
normalizer = Normalize(min_val, max_val)
im = cm.ScalarMappable(norm=normalizer)

for i, ax in enumerate(axes.flat):
    agent = model.schedule.agents[i]
    ax.imshow(agent.qmatrix, cmap=cmap, norm=normalizer)
    ax.set_title(str(i))
    if i + 1 == n_firms:
        break

plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.4, hspace=0.6)

fig.colorbar(im, ax=axes.ravel().tolist())
plt.show()

for i in range(int(np.ceil(n_firms / 3)) * 3 - n_firms):
    axes[int(np.ceil(n_firms / 3)) - 1, 2 - i].remove()

mean_profit = plt.plot(np.mean(profit_hist, axis=1))

fig = plt.figure()
mean_profit = fig.add_subplot()
mean_profit.plot(np.mean(profit_hist, axis=1))
mean_profit.set_title('Mean Profit')
