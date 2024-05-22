import collusionModel as model
from datetime import datetime
import numpy as np
import torch as T
from torch.utils.tensorboard import SummaryWriter
from agents import Firm
from generatePlots import generate, plot_alphas

seed = 123

schedule = []
T.manual_seed(seed)
np.random.seed(seed)

args = {'gamma': 0.9,
        'tau': 0.005,
        'alpha': 0.3,
        'policy': 'Gaussian',
        'target_update_interval': 1,
        'automatic_entropy_tuning': True,
        'cuda': T.device("cuda:0" if T.cuda.is_available() else "cpu"),
        'hidden_size': 256,
        'lr': 0.0003,
        'replay_size': 5 * 10 ** 5,
        'start_steps': 10000,
        'batch_size': 256,
        'updates_per_step': 1,
        'firms': 2,
        'game_length': 10 ** 6,
        'runs': 1,
        'demand_scale': 1
        }

print(args['cuda'])

env = model.CollusionModelSimultaneous(args['demand_scale'])

prices = np.zeros((args['game_length'], args['firms']))
demands = np.zeros((args['game_length'], args['firms']))
profits = np.zeros((args['game_length'], args['firms']))
alphas = np.zeros((args['game_length'] * args['updates_per_step'], args['firms']))


for agent in range(args['firms']):
    a = Firm(agent, env.state_space, env.action_space, seed, args)
    schedule.append(a)

#Tensorboard
writer = SummaryWriter('collusion/pythonFiles/runs/{}_SAC_{}_{}_{}'.format(datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), env.name, args['policy'], "autotune" if args['automatic_entropy_tuning'] else ""))


start = datetime.now()

# Training Loop
updates = 0

state = env.initialize()[0]
actions = np.zeros(args['firms'])

done = False
for episode in range(args['game_length']):

    if args['start_steps'] > episode:
        actions = env.sample_action()[0]  # Sample random action
    else:
        for agent in schedule:
            actions[agent.unique_id] = agent.act(state)  # Sample action from policy

    if len(schedule[0].memory) > args['batch_size']:
        for agent in schedule:
            # Number of updates per step in environment
            for i in range(args['updates_per_step']):
                # Update parameters of all the networks
                critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha = agent.update(agent.memory, args['batch_size'], updates)

                writer.add_scalar('loss/critic_1', critic_1_loss, updates)
                writer.add_scalar('loss/critic_2', critic_2_loss, updates)
                writer.add_scalar('loss/policy', policy_loss, updates)
                writer.add_scalar('loss/entropy_loss', ent_loss, updates)
                writer.add_scalar('entropy_temprature/alpha', alpha, updates)
                alphas[updates, agent.unique_id] = alpha
                updates += 1 if agent.unique_id % args['firms'] == 0 else 0

    next_state, demand, reward = env.step(actions)
    prices[episode, :] = actions
    demands[episode, :] = demand
    profits[episode, :] = reward

    for agent in schedule:
        agent.memory.push(state.reshape(-args['firms']), np.array([actions[agent.unique_id]]), reward[agent.unique_id], next_state.reshape(-args['firms']), done) # Append transition to memory

    state = next_state

end = datetime.now()
print('Start time was: {}'.format(start))
print('End time was: {}'.format(end))
print('Execution time was {}'.format(datetime.now() - start))

generate(args['firms'], args['runs'], prices, demands, profits)
plot_alphas(alphas)

np.save('collusion/outData/price_{}_{}_{}.npy'.format(args['firms'], args['runs'], args['demand_scale']), prices)
np.save('collusion/outData/demand_{}_{}_{}.npy'.format(args['firms'], args['runs'], args['demand_scale']), demands)
np.save('collusion/outData/profit_{}_{}_{}.npy'.format(args['firms'], args['runs'], args['demand_scale']), profits)
