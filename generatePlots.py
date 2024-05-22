import matplotlib.pyplot as plt
import numpy as np


def generate(n_firms, runs, prices, demands, profits):
    mean_price = []
    for agent in range(n_firms):
        mean_price.append(np.mean(prices[:, np.arange(agent, prices.shape[1], n_firms)], axis=1))
    mean_price = np.transpose(np.asarray(mean_price))

    mean_demand = []
    for agent in range(n_firms):
        mean_demand.append(np.mean(demands[:, np.arange(agent, demands.shape[1], n_firms)], axis=1))
    mean_demand = np.transpose(np.asarray(mean_demand))

    mean_profit = []
    for agent in range(n_firms):
        mean_profit.append(np.mean(profits[:, np.arange(agent, profits.shape[1], n_firms)], axis=1))
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

    plt.savefig('collusion/figures/sac_results_{}_{}.png'.format(n_firms, runs))


def plot_alphas(alphas):
    labels = ['Firm {}'.format(i) for i in np.arange(alphas.shape[1]) + 1]
    fig, ax = plt.subplots()
    ax.plot(alphas, label=labels)
    ax.legend(labels)

    plt.savefig('collusion/figures/alphas_{}.png'.format(alphas.shape[1]))
