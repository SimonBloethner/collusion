import numpy as np
from EconFunctions import CollusionModelSimultaneous
from datetime import datetime
import matplotlib.pyplot as plt

np.random.seed(2)

runs = 1
steps = 1000 * 1000
n_firms = 2
memory = 1
state_space = n_firms * memory
action_space = 1
multirun = True
load_checkpoint = False
demand_scale = 100

price_hist = np.zeros((steps, n_firms * (runs if multirun else 1)))
profit_hist = np.zeros((steps, n_firms * (runs if multirun else 1)))
demand_hist = np.zeros((steps, n_firms * (runs if multirun else 1)))

model = CollusionModelSimultaneous(n_firms, state_space, action_space, steps, run=0, load_checkpoint=load_checkpoint,demand_scale=demand_scale)

start = datetime.now()

for run in range(runs):
    model.__init__(n_firms, state_space, action_space, steps, run=run, load_checkpoint=load_checkpoint, demand_scale=demand_scale)
    for step in range(2, steps):
        model.step()

    model.transcribe(price_hist, demand_hist, profit_hist)


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

plt.savefig('collusion/figures/sac_results_{}_{}.png'.format(n_firms, runs))

np.save('collusion/outData/price_{}_{}_{}.npy'.format(n_firms, runs, demand_scale), price_hist)
np.save('collusion/outData/demand_{}_{}_{}.npy'.format(n_firms, runs, demand_scale), demand_hist)
np.save('collusion/outData/profit_{}_{}_{}.npy'.format(n_firms, runs, demand_scale), profit_hist)

end = datetime.now()
print('Start time was: {}'.format(start))
print('End time was: {}'.format(end))
print('Execution time was {}'.format(datetime.now() - start))
