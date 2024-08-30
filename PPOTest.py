import torch.nn as nn
from torch.distributions import MultivariateNormal
from torch.distributions import Categorical
from tqdm import tqdm
import matplotlib.pyplot as plt
import os

import torch
import numpy as np


np.random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed(1)

path_ = os.getcwd()
local = path_.find('Simon') > 0
if local:
    path_ = '/Users/Simon/Documents/Projects/EWF/Research/PhD/Ergodicity Economics/IOxEE'
else:
    path_ = 'collusion'

device = 'cpu'


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, has_continuous_action_space, action_std_init=0.001):
        super(ActorCritic, self).__init__()

        self.has_continuous_action_space = has_continuous_action_space

        if has_continuous_action_space:
            self.action_dim = action_dim
            self.action_var = torch.full((action_dim,), action_std_init * action_std_init).to(device)
        # actor
        if has_continuous_action_space:
            self.actor = nn.Sequential(nn.Linear(state_dim, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh(),
                nn.Linear(64, action_dim), nn.Tanh())
        else:
            self.actor = nn.Sequential(nn.Linear(state_dim, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh(),
                nn.Linear(64, action_dim), nn.Softmax(dim=-1))
        # critic
        self.critic = nn.Sequential(nn.Linear(state_dim, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, 1))

    def act(self, state):
        if self.has_continuous_action_space:
            action_mean = self.actor(state)
            cov_mat = torch.diag(self.action_var).unsqueeze(dim=0)
            dist = MultivariateNormal(action_mean, cov_mat)
        else:
            action_probs = self.actor(state)
            dist = Categorical(action_probs)

        action = dist.sample()
        # action = torch.sigmoid(action)
        action = (torch.tanh(action) + 1) / 2
        action_logprob = dist.log_prob(action)
        state_val = self.critic(state)

        return action.detach(), action_logprob.detach(), state_val.detach()



class Firm:
    def __init__(self, number, state_dim, action_dim, has_continuous_action_space, action_std_init=0.6):
        self.id = number

        self.policy = ActorCritic(state_dim, action_dim, has_continuous_action_space, action_std_init).to(device)

        self.policy_old = ActorCritic(state_dim, action_dim, has_continuous_action_space, action_std_init).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.load(checkpoint_path=checkpoint_path +'/agent_' + '1' + '.pth') # str(self.id)

    def select_action(self, state):
        with torch.no_grad():
            state = torch.FloatTensor(state).to(device)
            action, action_logprob, state_val = self.policy_old.act(state)

        return action.detach().cpu().numpy().flatten()

    def load(self, checkpoint_path):
        self.policy_old.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        self.policy.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))


class CollusionModel():
    def __init__(self, n, iterations, shares):
        self.num_agents = n
        self.period = 0
        self.schedule = [Firm(firm, state_dim, action_dim, has_continuous_action_space,
                action_std) for firm in range(self.num_agents)]
        self.prices = np.zeros((iterations, n))
        self.demand = np.zeros((iterations, n))
        self.profits = np.zeros((iterations, n))
        self.shares = shares

    def initialize(self, iterations, n):
        self.prices = np.zeros((iterations, n))
        self.demand = np.zeros((iterations, n))
        self.profits = np.zeros((iterations, n))
        self.period = 0

    def step(self):

        action = []
        for a in range(len(self.schedule)):
            state = [self.prices[self.period - 1, a]]
            mask = np.ones(self.num_agents, dtype=bool)
            mask[a] = False
            p_ = np.max(self.prices[self.period - 1, mask])
            p__ = np.max(self.prices[self.period - 2, mask])
            # state_ = [[self.prices[self.period - 1, a_], self.prices[self.period - 2, a_]] for a_ in np.arange(self.num_agents)[np.r_[0:a, a+1:self.num_agents]]]
            # state_ = sum(state_, [])
            state_ = [p_, p__]
            state.extend(state_)
            state.append(self.shares[self.period - 1, a])
            action.append(self.schedule[a].select_action(state)[0])
        done = False

        self.prices[self.period, :] = action

        self.demand[self.period, :] = demand(action, shares[self.period, :])
        self.profits[self.period, :] = profit(action, shares[self.period, :])

        self.period += 1


def demand(p, share):
    d = [1 - p[firm] * (1 - share[firm]) + 0.5 * np.max(p[:firm] + p[firm+1:]) for firm in range(len(p))]
    return d


def profit(p, share):
    pi = np.array(p) * demand(p, share)
    return pi


its = 10 ** 2
num_agents = 2
has_continuous_action_space = True  # continuous action space; else discrete
shares = np.arange(0.01, 1, 0.01)
shares = np.hstack([shares, shares[::-1]])
# shares = np.hstack([shares.reshape(-1, 1), (1 - shares.reshape(-1, 1)) * shares[::-1].reshape(-1, 1), (1 - shares.reshape(-1, 1)) * (1 - shares[::-1].reshape(-1, 1))])
# shares = np.vstack([shares, shares[:, [2, 0, 1]], shares[:, [1, 2, 0]]])
shares = np.hstack([shares.reshape(-1, 1), 1 - shares.reshape(-1, 1)])
episodes = shares.shape[0] * its
shares = np.tile(shares, (its, 1))



action_std = 0.001  # starting std for action distribution (Multivariate Normal)
action_std_decay_rate = 0.05  # linearly decay action_std (action_std = action_std - action_std_decay_rate)

random_seed = 0  # set random seed if required (0 = no random seed)
checkpoint_path = '{}/{}/models'.format(path_, 'programs' if local else 'pythonFiles')
state_dim = 4

# action space dimension
if has_continuous_action_space:
    action_dim = 1
else:
    action_dim = 1

iterations = 1

model = CollusionModel(n=num_agents, iterations=episodes, shares=shares)
strg_price = np.empty((episodes, num_agents, iterations))
strg_demand = np.empty((episodes, num_agents, iterations))
strg_profit = np.empty((episodes, num_agents, iterations))

for it in range(iterations):
    for episode in tqdm(range(episodes)):
        model.step()
    strg_price[:, :, it] = model.prices
    strg_demand[:, :, it] = model.demand
    strg_profit[:, :, it] = model.profits

    model.initialize(iterations=episodes, n=num_agents)


slice_ = iterations - 1
labels = ['Firm {}'.format(i) for i in range(num_agents)]
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True)
ax1.plot(strg_price[:, :, slice_], label=labels)
ax1.title.set_text('Price')
ax2.plot(strg_demand[:, :, slice_], label=labels)
ax2.title.set_text('Demand')
ax3.plot(strg_profit[:, :, slice_], label=labels)
ax3.title.set_text('Profit')
fig.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.4, hspace=0.4)
fig.legend(labels, loc='upper right', bbox_transform=fig.transFigure)

plt.savefig('{}/figures/SimPPO.png'.format(path_))
