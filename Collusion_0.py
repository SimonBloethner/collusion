from mesa import Agent, Model
from mesa.time import RandomActivation
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

np.random.seed(2)
n_firms = 10
steps = 200
max_demand = 100

price_hist = np.zeros((steps, n_firms))
profit_hist = np.zeros((steps, n_firms))
demand_hist = np.zeros((steps, n_firms))


def logistic(x, L, k, x_0):
    return L / (1 + np.exp(-k * (x - x_0)))


class Firm(Agent):

    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.price = np.random.uniform(1, 50, 1)
        self.profit = np.zeros((steps, 1))
        self.demand = np.zeros((steps, 1))
        self.price_list = np.zeros((steps, 1))

    def step(self) -> None:
        if model.period < 2:
            self.price = np.random.uniform(1, 50, 1)
            self.price_list[model.period] = self.price
        else:
            if model.mode == 'binary':
                self.update_binary()
            elif model.mode == 'proportional':
                self.update_proportinal()

        self.price_list[model.period] = self.price
        model.price_list[model.period, self.unique_id] = self.price

    def update_binary(self):
        change = self.profit[model.period - 1] / self.profit[model.period - 2]
        if self.profit[model.period - 1] - self.profit[model.period - 2] > 0 and self.price_list[model.period - 1] - self.price_list[model.period - 2] > 0:
            self.price += 2 * (logistic(change, 2, 7, 1) - 1)
        elif self.profit[model.period - 1] - self.profit[model.period - 2] > 0 and self.price_list[model.period - 1] - self.price_list[model.period - 2] < 0:
            self.price -= 2 * (logistic(change, 2, 7, 1) - 1)
        elif self.profit[model.period - 1] - self.profit[model.period - 2] < 0 and self.price_list[model.period - 1] - self.price_list[model.period - 2] > 0:
            self.price -= 2 * (logistic(change, 2, -7, 1) - 1)
        elif self.profit[model.period - 1] - self.profit[model.period - 2] < 0 and self.price_list[model.period - 1] - self.price_list[model.period - 2] < 0:
            self.price += 2 * (logistic(change, 2, -7, 1) - 1)

        if self.price < 0:
            self.price = 0

    def update_proportinal(self):
        if self.profit[model.period - 2] == 0 or self.profit[model.period - 1] == 0:
            self.price *= 0.98
        else:
            change = self.profit[model.period - 1] / self.profit[model.period - 2]

            if self.profit[model.period - 1] - self.profit[model.period - 2] > 0 and self.price_list[model.period - 1] - self.price_list[model.period - 2] > 0:
                self.price *= logistic(change, 2, 7, 1)
            elif self.profit[model.period - 1] - self.profit[model.period - 2] > 0 and self.price_list[model.period - 1] - self.price_list[model.period - 2] < 0:
                self.price *= logistic(change, 2, -7, 1)
            elif self.profit[model.period - 1] - self.profit[model.period - 2] < 0 and self.price_list[model.period - 1] - self.price_list[model.period - 2] > 0:
                self.price *= logistic(change, 2, 7, 1)
            elif self.profit[model.period - 1] - self.profit[model.period - 2] < 0 and self.price_list[model.period - 1] - self.price_list[model.period - 2] < 0:
                self.price *= logistic(change, 2, -7, 1)

            # self.price *= change
            # self.price *= abs(self.profit[model.period - 1] - self.profit[model.period - 2])/ self.profit[model.period - 2]
            # self.price *= 1 + (np.log(change) if change > 1 else -np.log(change))
            # self.price *= 0.5 + logistic(change, 10, 1)


class CollusionModel(Model):

    def __init__(self, N, mode):
        self.num_agents = N
        self.mode = mode
        self.period = 0
        self.max_demand = max_demand
        self.demand_list = []
        self.price_list = np.vstack([np.random.uniform(1, 50, n_firms).reshape(1, n_firms), np.zeros((steps - 1, n_firms))])
        # Create agents
        self.schedule = RandomActivation(self)
        for i in range(self.num_agents):
            a = Firm(i, self)
            self.schedule.add(a)

    def step(self):
        self.schedule.step()
        self.quantity_allocation()
        for a in self.schedule.agents:
            a.demand[self.period] = self.max_demand - self.price_list[self.period, a.unique_id] + (0.1 / 9) * (np.sum(self.price_list[self.period, :]) - self.price_list[self.period, a.unique_id])
            if a.demand[self.period] < 0:
                a.demand[self.period] = 0
            a.profit[self.period] = a.demand[self.period] * a.price_list[self.period]

        self.period += 1
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


model = CollusionModel(n_firms, mode='proportional')

for i in tqdm(range(steps)):
    model.step()

fig = plt.figure(figsize=plt.figaspect(0.5))
profit_plt = fig.add_subplot(3, 1, 1)
profit_plt.plot(profit_hist)
profit_plt.set_title('Profits')

price_plt = fig.add_subplot(3, 1, 2)
price_plt.plot(price_hist)
price_plt.set_title('Prices')

demand_plt = fig.add_subplot(3, 1, 3)
demand_plt.plot(demand_hist)
demand_plt.set_title('Demand')

plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.4, hspace=0.6)

agent_view = plt.figure(figsize=plt.figaspect(0.5))
for agent in range(n_firms):
    agent_plt = agent_view.add_subplot(int(np.ceil(n_firms / 3)), 3, agent + 1)
    agent_plt.plot(profit_hist[:, agent]), agent_plt.plot(price_hist[:, agent]), agent_plt.plot(demand_hist[:, agent])
    agent_plt.set_title('{}'.format(agent))


plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.4, hspace=0.6)
