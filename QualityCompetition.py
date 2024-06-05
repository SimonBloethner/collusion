import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import random


np.random.seed(0)
random.seed(0)
torch.manual_seed(0)


class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.price_mean = nn.Linear(64, action_dim)
        self.price_log_std = nn.Linear(64, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        price_mean = self.price_mean(x)
        price_log_std = self.price_log_std(x)
        return price_mean, price_log_std


class ValueNetwork(nn.Module):
    def __init__(self, state_dim):
        super(ValueNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.value = nn.Linear(64, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        value = self.value(x)
        return value


class Firm:
    def __init__(self, number, state_dim, action_dim, lr=1e-5, gamma=0.99, eps_clip=0.2):
        self.id = number
        self.policy = PolicyNetwork(state_dim, action_dim)
        self.value_function = ValueNetwork(state_dim)
        self.optimizer_policy = optim.Adam(self.policy.parameters(), lr=lr)
        self.optimizer_value = optim.Adam(self.value_function.parameters(), lr=lr)
        self.gamma = gamma
        self.clip_epsilon = eps_clip
        self.states, self.actions, self.log_probs, self.rewards, self.values = [], [], [], [], []
        self.entropy_coef = 0.05

    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        price_mean, price_log_std = self.policy(state)
        price_std = price_log_std.exp()
        dist = Normal(price_mean, price_std)
        action = dist.sample()
        action = torch.sigmoid(action)
        return action.item(), dist.log_prob(action).sum(dim=-1), dist.entropy().sum(dim=-1)

    def normalize_rewards(self, rewards):
        rewards = torch.tensor(rewards, dtype=torch.float32)
        return (rewards - rewards.mean()) / (rewards.std() + 1e-8)

    def update(self, rewards, log_probs, states, actions, values):
        rewards = self.normalize_rewards(rewards).tolist()
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + self.gamma * G
            returns.insert(0, G)
        returns = torch.tensor(returns, dtype=torch.float32)
        states = torch.stack(states)
        actions = torch.tensor(actions, dtype=torch.float32)
        old_log_probs = torch.stack(log_probs).detach()

        advantages = returns - values.detach()

        for _ in range(10):  # PPO usually performs multiple epochs of updates
            # Recompute log_probs and values
            price_mean, price_log_std = self.policy(states)
            price_std = price_log_std.exp()
            dist = Normal(price_mean, price_std)
            new_log_probs = dist.log_prob(actions).sum(dim=-1)
            entropy = dist.entropy().sum(dim=-1)

            # Compute ratios
            ratios = torch.exp(new_log_probs - old_log_probs)

            # Compute surrogate losses
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            # Compute value loss
            new_values = self.value_function(states)
            value_loss = (returns - new_values).pow(2).mean()

            # Update policy
            self.optimizer_policy.zero_grad()
            policy_loss.backward(retain_graph=True)
            self.optimizer_policy.step()

            # Update value function
            self.optimizer_value.zero_grad()
            value_loss.backward()
            self.optimizer_value.step()


class CollusionModel():
    def __init__(self, n, iterations):
        self.num_agents = n
        self.period = 3
        self.demand_list = []
        self.schedule = [Firm(number=firm, action_dim=1, state_dim=3) for firm in range(self.num_agents)]
        self.prices = np.zeros((iterations + 2, n))
        self.demand = np.zeros((iterations + 2, n))
        self.profits = np.zeros((iterations + 2, n))
        self.device = 'cpu'

    def step(self):
        if self.period == 10159:
            print('!')
        a = self.period % 2
        done = False
        state = [self.prices[self.period - 3, a], self.prices[self.period - 2, 0 ** a], self.prices[self.period - 4, 0 ** a]]

        action, log_prob, entropy = self.schedule[a].select_action(state)

        self.prices[[self.period - 2, self.period - 1], a] = action

        self.demand[self.period - 2, :] = demand(self.prices[self.period - 2, 0], self.prices[self.period - 2, 1])
        self.profits[self.period - 2, :] = profit(self.prices[self.period - 2, 0], self.prices[self.period - 2, 1])

        next_state = np.array([action, self.prices[self.period - 2, 0 ** a], self.prices[self.period - 1, 0 ** a]])

        value = self.schedule[a].value_function(torch.FloatTensor(state).unsqueeze(0))

        reward = self.profits[self.period - 2, a]

        self.schedule[a].states.append(torch.FloatTensor(state))
        self.schedule[a].actions.append(action)
        self.schedule[a].log_probs.append(log_prob)
        self.schedule[a].rewards.append(reward)
        self.schedule[a].values.append(value)

        if len(self.schedule[a].rewards) > 10:
            self.schedule[a].update(self.schedule[a].rewards, self.schedule[a].log_probs, self.schedule[a].states, self.schedule[a].actions, torch.cat(self.schedule[a].values).squeeze())
            self.schedule[a].states, self.schedule[a].actions, self.schedule[a].log_probs, self.schedule[a].rewards, self.schedule[a].values = [], [], [], [], []

        self.period += 1


def demand(p_i, p_j):
    p = np.array([[p_i, p_j], [p_j, p_i]])
    d = 1 - p + 0.5 * p
    return d[:, 0]


def profit(p_i, p_j):
    pi = p_i * demand(p_i, p_j)
    return pi


iterations = 10 ** 5
model = CollusionModel(n=2, iterations=iterations)

for episode in tqdm(range(iterations)):
    model.step()

labels = ['Firm 1', 'Firm 2']
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True)
ax1.plot(model.prices[2:-1], label=['Firm 1', 'Firm 2'])
ax1.title.set_text('Price')
ax2.plot(model.demand[2:-1], label=['Firm 1', 'Firm 2'])
ax2.title.set_text('Demand')
ax3.plot(model.profits[2:-1], label=['Firm 1', 'Firm 2'])
ax3.title.set_text('Profit')
fig.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.4, hspace=0.4)
fig.legend(labels, loc='upper right', bbox_transform=fig.transFigure)
