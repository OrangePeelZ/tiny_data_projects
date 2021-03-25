import numpy as np
import gym
import torch
import torch.optim as optim
from utils import select_action, ReplayMemory, optimize_model, DQN, save_model, save_rewards
# import some constants
from utils import N_STEPS_TIMEOUT, TARGET_UPDATE, MEMORY_CAPACITY, BATCH_SIZE, EPS_START, EPS_END
import random
from itertools import count
from multiprocessing import Pool

# grid search parameters
EPS_DECAY = [1000, 10000, 50000]
GAMMA = [0.8, 0.99, 0.999]
LR = [0.01, 0.001, 0.0001]
NETWORK = {'simple': (16, 8),
           'medium': (64, 16),
           'complex': (256, 128)}

# hyper parameters
N_EPISODES = 1000


def train(eps_decay, gamma, lr, network, seed=131):
    id = 'LunarLander-v2'
    env = gym.make(id).unwrapped
    n_actions = env.action_space.n
    n_states = env.observation_space.shape[0]
    # set seed
    random.seed(seed)
    env.seed(seed)

    # initiate the network
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if network not in NETWORK.keys():
        raise ValueError('Network key not existed!')

    fc1_unit, fc2_unit = NETWORK.get(network)
    policy_net = DQN(state_size=n_states,
                     action_size=n_actions,
                     fc1_unit=fc1_unit,
                     fc2_unit=fc2_unit,
                     seed=131).to(device)
    target_net = DQN(state_size=n_states,
                     action_size=n_actions,
                     fc1_unit=fc1_unit,
                     fc2_unit=fc2_unit,
                     seed=1).to(device)
    target_net.load_state_dict(policy_net.state_dict())

    # initiate the memory replayer and optimizer
    memory = ReplayMemory(MEMORY_CAPACITY)
    # optimizer = optim.RMSprop(policy_net.parameters())
    optimizer = optim.Adam(policy_net.parameters(), lr=lr)

    # initiate the global steps
    steps_done = 0
    # Here my watch started
    rewards = []
    for i_episode in range(N_EPISODES):
        cumulative_reward = 0
        state = env.reset()
        state = torch.tensor([state])
        for t in count():
            if t > N_STEPS_TIMEOUT:
                break
            action, steps_done = select_action(state=state,
                                               policy_net=policy_net,
                                               n_actions=n_actions,
                                               steps_done=steps_done,
                                               device=device,
                                               eps_end=EPS_END,
                                               eps_start=EPS_START,
                                               eps_decay=eps_decay)

            state_next, reward, done, _ = env.step(action.item())
            # env.render()
            cumulative_reward = cumulative_reward + reward
            # convert it to tensor
            state_next = torch.tensor([state_next], device=device)
            reward = torch.tensor([reward], device=device, dtype=torch.float32)
            memory.push(state,
                        action,
                        state_next,
                        reward)
            state = state_next

            # every step update the weights in the policy net
            optimize_model(memory=memory,
                           batch_size=BATCH_SIZE,
                           device=device,
                           policy_net=policy_net,
                           target_net=target_net,
                           optimizer=optimizer,
                           gamma=gamma)

            if done:
                break

        rewards.append(cumulative_reward)

        # update the target net after a while
        if i_episode % TARGET_UPDATE == 0:
            # If want the soft update the weights
            #         soft_update(local_model=policy_net, target_model=target_net, tau=TAU)
            target_net.load_state_dict(policy_net.state_dict())

        if np.min(rewards[-5:]) >= 200:
            break

    # save the rewards
    rewards_path = 'training_rewards_{lr}_{eps_decay}_{gamma}_{network}.pkl'.format(lr=lr, eps_decay=eps_decay,
                                                                                    gamma=gamma,
                                                                                    network=network)
    save_rewards(rewards=rewards, path=rewards_path, option='training_rewards')

    # save the policy net
    model_path = 'model_{lr}_{eps_decay}_{gamma}_{network}.pt'.format(lr=lr, eps_decay=eps_decay, gamma=gamma,
                                                                      network=network)
    save_model(model=policy_net, path=model_path)
    print ("Finished parameter combo: {params}".format(params=[eps_decay, gamma, lr, network]))


if __name__ == '__main__':
    # parallelize the training
    hyper_params = [(eps_decay, gamma, lr, network) for eps_decay in EPS_DECAY
                    for gamma in GAMMA
                    for lr in LR
                    for network in NETWORK.keys()]
    pool = Pool(6)
    pool.starmap(train, hyper_params)
    pool.close()
    # train(lr=0.001,eps_decay=10000, gamma=0.99, network='simple')
    # with 2 workers 733.70s user 46.49s system 701% cpu 1:51.21 total
    # with 1 worker  759.59s user 18.21s system 458% cpu 2:49.74 total

