from sac_ import SAC
from replay_memory import ReplayMemory


class Firm:
    def __init__(self, unique_id, state_space, action_space, seed, args):
        self.unique_id = unique_id
        self.networks = SAC(state_space, action_space, args)
        self.memory = ReplayMemory(args['replay_size'], seed)

    def update(self, memory, batch_size, updates):
        return self.networks.update_parameters(memory, batch_size, updates)

    def act(self, state, evaluate=False):
        return self.networks.select_action(state, evaluate)

    def remember(self, state, action, reward, next_state, mask):
        self.memory.push(state, action, reward, next_state, mask)
