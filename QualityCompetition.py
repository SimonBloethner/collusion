import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gym


class PolicyNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.price_mean = nn.Linear(128, 1)
        self.price_log_std = nn.Linear(128, 1)
        # self.quality_mean = nn.Linear(128, 1)
        # self.quality_log_std = nn.Linear(128, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        price_mean = self.price_mean(x)
        price_log_std = self.price_log_std(x)
        # quality_mean = self.quality_mean(x)
        # quality_log_std = self.quality_log_std(x)
        return price_mean, price_log_std    # , quality_mean, quality_log_std


class ValueNetwork(nn.Module):
    def __init__(self, state_size):
        super(ValueNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.value = nn.Linear(128, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.value(x)


class PPOAgent:
    def __init__(self, state_size, action_size, lr=3e-4, gamma=0.99, clip_epsilon=0.2):
        self.policy_net = PolicyNetwork(state_size, action_size)
        self.value_net = ValueNetwork(state_size)
        self.optimizer_policy = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.optimizer_value = optim.Adam(self.value_net.parameters(), lr=lr)
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self.state_size = state_size
        self.action_size = action_size

    def select_action(self, state):
        state = torch.FloatTensor(state)
        # price_mean, price_log_std, quality_mean, quality_log_std = self.policy_net(state)
        price_mean, price_log_std = self.policy_net(state)
        price_std = torch.exp(price_log_std)
        # quality_std = torch.exp(quality_log_std)
        price_dist = torch.distributions.Normal(price_mean, price_std)
        # quality_dist = torch.distributions.Normal(quality_mean, quality_std)
        price_action = price_dist.sample()
        # quality_action = quality_dist.sample()
        # action = torch.cat([price_action, quality_action], dim=-1)
        action = torch.cat([price_action], dim=-1)
        # log_prob = price_dist.log_prob(price_action) + quality_dist.log_prob(quality_action)
        log_prob = price_dist.log_prob(price_action)  # + quality_dist.log_prob(quality_action)
        return action, log_prob

    def compute_gae(self, rewards, values, next_values, dones):
        advantages = []
        gae = 0
        for i in reversed(range(len(rewards))):
            delta = rewards[i] + self.gamma * next_values[i] * (1 - dones[i]) - values[i]
            gae = delta + self.gamma * gae
            advantages.insert(0, gae)
        return advantages

    def update(self, trajectories):
        states = torch.FloatTensor([t['state'] for t in trajectories])
        actions = torch.FloatTensor([t['action'] for t in trajectories])
        log_probs = torch.FloatTensor([t['log_prob'] for t in trajectories])
        rewards = torch.FloatTensor([t['reward'] for t in trajectories])
        next_states = torch.FloatTensor([t['next_state'] for t in trajectories])
        dones = torch.FloatTensor([t['done'] for t in trajectories])
        values = self.value_net(states).squeeze()
        next_values = self.value_net(next_states).squeeze()

        advantages = self.compute_gae(rewards, values, next_values, dones)
        advantages = torch.FloatTensor(advantages)

        for _ in range(10):  # PPO usually performs multiple epochs of updates
            new_log_probs = self.policy_net.get_log_probs(states, actions)
            ratios = torch.exp(new_log_probs - log_probs)
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = (rewards + self.gamma * next_values - values).pow(2).mean()

            self.optimizer_policy.zero_grad()
            policy_loss.backward()
            self.optimizer_policy.step()

            self.optimizer_value.zero_grad()
            value_loss.backward()
            self.optimizer_value.step()

    def store_trajectory(self, state, action, log_prob, reward, next_state, done):
        trajectory = {
            'state': state,
            'action': action,
            'log_prob': log_prob,
            'reward': reward,
            'next_state': next_state,
            'done': done
        }
        return trajectory

# Example usage
env = gym.make('YourCustomEnv-v0')
agent = PPOAgent(state_size=4, action_size=2)

for episode in range(1000):
    state = env.reset()
    episode_rewards = []
    trajectories = []
    done = False
    while not done:
        action, log_prob = agent.select_action(state)
        next_state, reward, done, _ = env.step(action.numpy())
        trajectory = agent.store_trajectory(state, action, log_prob, reward, next_state, done)
        trajectories.append(trajectory)
        episode_rewards.append(reward)
        state = next_state

    agent.update(trajectories)
    print(f'Episode {episode} reward: {sum(episode_rewards)}')
