import numpy as np
import matplotlib.pyplot as plt
from collusionModel import market_outcome
from matplotlib.colors import Normalize
import matplotlib.cm as cm


def global_econs(prices, rest, num, demand_scale):
    d = (1 * demand_scale) - prices + (1 / num) * rest
    p = d * prices
    return d, p


filename = '/Users/Simon/price_2_1_1.npy'

params = [i for i, letter in enumerate(filename) if letter == '_']

n_firms = int(filename[params[0] + 1])
runs = int(filename[params[1] + 1])
demand_scale = int(filename[params[2] + 1:filename.find('.')])

price_hist = np.load(filename)
demand_hist = np.zeros(price_hist.shape)
profit_hist = np.zeros(price_hist.shape)

for run in range(runs):
    for agent in range(n_firms):
        rest = np.sum(np.delete(price_hist[:, (run * n_firms):((run + 1) * n_firms)], agent, axis=1), axis=1)
        demand, profit = global_econs(price_hist[:, run + agent], rest, n_firms, demand_scale)
        demand_hist[:, run + agent] = demand
        profit_hist[:, run + agent] = profit

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
