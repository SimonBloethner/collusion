import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gym
import random
import numpy as np
import matplotlib.pyplot as plt
from mesa import Agent, Model
from mesa.time import BaseScheduler
from datetime import datetime

from EconFunctions import demand

steps = 1000 * 500
n_firms = 2
runs = 100
multirun = True
LEARNING_RATE = 0.0001
MEM_SIZE = 10000
BATCH_SIZE = 64
GAMMA = 0.95
EXPLORATION_MAX = 1.0
EXPLORATION_DECAY = 0.999
EXPLORATION_MIN = 0.001
state_space = 1
actions_space = np.linspace(0, 1, 6)

FC1_DIMS = 1024
FC2_DIMS = 512
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(DEVICE)

best_reward = 0
average_reward = 0
episode_number = []
average_reward_number = []

np.random.seed(66)
price_hist = np.zeros((steps, n_firms * (runs if multirun else 1)))
profit_hist = np.zeros((steps, n_firms * (runs if multirun else 1)))
demand_hist = np.zeros((steps, n_firms * (runs if multirun else 1)))


class Firm(Agent):

    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.price = np.random.uniform(0, 1)
        self.profit = np.zeros((steps, 1))
        self.demand = np.zeros((steps, 1))
        self.price_list = np.zeros((steps, 1))
        self.price_list[0:2] = np.random.uniform(0, 1, 2).reshape(2, 1)
        self.state = 0
        self.action = 0
        self.memory = ReplayBuffer()
        self.exploration_rate = EXPLORATION_MAX
        self.network = Network()

    def choose_action(self, observation):
        if random.random() < self.exploration_rate:
            return np.random.uniform(0, 1)

        state = torch.tensor(observation).float().detach()
        state = state.to(DEVICE)
        state = state.unsqueeze(0)
        q_values = self.network(state)
        self.action = actions_space[torch.argmax(q_values).item()]
        return self.action

    def learn(self):
        if self.memory.mem_count < BATCH_SIZE:
            return

        states, actions, rewards, states_, dones = self.memory.sample()
        states = torch.tensor(states, dtype=torch.float32).to(DEVICE)
        actions = torch.tensor(actions, dtype=torch.long).to(DEVICE)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(DEVICE)
        states_ = torch.tensor(states_, dtype=torch.float32).to(DEVICE)
        dones = torch.tensor(dones, dtype=torch.bool).to(DEVICE)
        batch_indices = np.arange(BATCH_SIZE, dtype=np.int64)

        q_values = self.network(states)
        next_q_values = self.network(states_)

        predicted_value_of_now = q_values[batch_indices, actions]
        predicted_value_of_future = torch.max(next_q_values, dim=1)[0]

        q_target = rewards + GAMMA * predicted_value_of_future

        loss = self.network.loss(q_target, predicted_value_of_now)
        self.network.optimizer.zero_grad()
        loss.backward()
        self.network.optimizer.step()

        self.exploration_rate *= EXPLORATION_DECAY
        self.exploration_rate = max(EXPLORATION_MIN, self.exploration_rate)

    def returning_epsilon(self):
        return self.exploration_rate


class CollusionModel(Model):
    def __init__(self, N, state_space, action_space):
        self.state_space = state_space
        self.action_space = action_space
        self.num_agents = N
        self.period = 2
        # self.max_demand = max_demand
        self.demand_list = []
        self.prices = np.zeros([steps, n_firms])

        # Create agents
        self.schedule = BaseScheduler(self)
        for i in range(self.num_agents):
            a = Firm(i, self)
            self.schedule.add(a)
        for a in self.schedule.agents:
            self.prices[0:2, a.unique_id] = a.price_list[0:2].reshape(2, )
        for a in self.schedule.agents:
            rest = np.mean(np.delete(self.prices[0:2, :], a.unique_id, axis=1), axis=1)  # Here is where we lose information due to rounding the mean.
            a.demand[0:2] = demand(a.price_list[0:2].reshape(2, ), rest, n_firms).reshape(2, 1)
            a.profit[0:2] = a.demand[0:2] * a.price_list[0:2]

    def step(self):
        for a in self.schedule.agents:
            rest = np.mean(np.delete(self.prices[self.period - 1, :], a.unique_id))  # Here is where we lose information due to rounding the mean.

            a.choose_action(rest)

        for a in self.schedule.agents:
            rest = np.mean(np.delete(self.prices[self.period, :], a.unique_id))
            a.demand[self.period] = demand(a.price_list[self.period], rest, n_firms)
            a.profit[self.period] = a.demand[self.period] * a.price_list[self.period]

        for a in self.schedule.agents:
            rest = np.mean(np.delete(self.prices[self.period - 1, :], a.unique_id))
            state_ = np.mean(np.delete(self.prices[self.period, :], a.unique_id))
            a.memory.add(rest, a.action, a.profit[self.period], state_, 0)
            a.learn()

        self.period += 1
        if self.period == steps:
            agent_id = 0
            for a in self.schedule.agents:
                for time in range(steps - 2, steps):
                    rest = np.mean(np.delete(self.prices[time, :], a.unique_id))  # Here is where we lose information due to rounding the mean.
                    a.demand[time] = demand(a.price_list[time], rest, n_firms)
                    a.profit[time] = a.demand[time] * a.price_list[time]

                price_hist[:, agent_id + j * n_firms] = a.price_list.reshape(-1)
                demand_hist[:, agent_id + j * n_firms] = a.demand.reshape(-1)
                profit_hist[:, agent_id + j * n_firms] = a.profit.reshape(-1)
                agent_id += 1


class Network(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.input_shape = state_space
        self.action_space = actions_space.shape[0]

        self.fc1 = nn.Linear(self.input_shape, FC1_DIMS)
        self.fc2 = nn.Linear(FC1_DIMS, FC2_DIMS)
        self.fc3 = nn.Linear(FC2_DIMS, self.action_space)

        self.optimizer = optim.Adam(self.parameters(), lr=LEARNING_RATE)
        self.loss = nn.MSELoss()
        self.to(DEVICE)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x


class ReplayBuffer:
    def __init__(self):
        self.mem_count = 0
        self.states = np.zeros((MEM_SIZE, state_space), dtype=np.float32)
        self.actions = np.zeros(MEM_SIZE, dtype=np.int64)
        self.rewards = np.zeros(MEM_SIZE, dtype=np.float32)
        self.states_ = np.zeros((MEM_SIZE, state_space), dtype=np.float32)
        self.dones = np.zeros(MEM_SIZE)

    def add(self, state, action, reward, state_, done):
        mem_index = self.mem_count % MEM_SIZE

        self.states[mem_index] = state
        self.actions[mem_index] = action
        self.rewards[mem_index] = reward
        self.states_[mem_index] = state_
        self.dones[mem_index] = 1 - done

        self.mem_count += 1

    def sample(self):
        MEM_MAX = min(self.mem_count, MEM_SIZE)
        batch_indices = np.random.choice(MEM_MAX, BATCH_SIZE, replace=True)

        states = self.states[batch_indices]
        actions = self.actions[batch_indices]
        rewards = self.rewards[batch_indices]
        states_ = self.states_[batch_indices]
        dones = self.dones[batch_indices]

        return states, actions, rewards, states_, dones


model = CollusionModel(n_firms, state_space=state_space, action_space=actions_space.shape[0])

start = datetime.now()

if multirun:
    for j in range(runs):
        model.__init__(n_firms, state_space=state_space, action_space=actions_space.shape[0])
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
