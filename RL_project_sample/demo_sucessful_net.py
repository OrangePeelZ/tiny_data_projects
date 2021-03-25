import numpy as np
import gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from utils import select_action, ReplayMemory, optimize_model, soft_update,save_model,save_rewards, DQN
import random
import math
from itertools import count


# Global parameters
EPS_START = 0.9
EPS_END = 0.1
EPS_DECAY = 10000
GAMMA = 0.99
TARGET_UPDATE = 10
MEMORY_CAPACITY = 10000
BATCH_SIZE = 64
TAU = 0.1
N_STEPS_TIMEOUT = 20000

# hyper parameters
LR = 0.001
N_EPISODES = 1000


# The network


if __name__ == '__main__':
    # initiate the environment
    id = 'LunarLander-v2'
    env = gym.make(id).unwrapped
    n_actions = env.action_space.n
    n_states = env.observation_space.shape[0]
    # set seed
    random.seed(131)
    env.seed(131)

    # initiate the network
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    policy_net = DQN(state_size=n_states,
                     action_size=n_actions,
                     seed=131).to(device)
    target_net = DQN(state_size=n_states,
                     action_size=n_actions,
                     seed=1).to(device)
    target_net.load_state_dict(policy_net.state_dict())

    # initiate the memory replayer and optimizer
    memory = ReplayMemory(MEMORY_CAPACITY)
    optimizer = optim.Adam(policy_net.parameters(), lr=LR)

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
                                               eps_decay=EPS_DECAY)

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
                           gamma=GAMMA)

            if done:
                break

        rewards.append(cumulative_reward)
        rate = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)
        print('cumulative reward for episode {n_ep} is {cum_reward}; With the epsilon: {eps}'. \
              format(n_ep=i_episode,
                     cum_reward=cumulative_reward,
                     eps=rate))

        # update the target net after a while
        if i_episode % TARGET_UPDATE == 0:
            # If want the soft update the weights
            #         soft_update(local_model=policy_net, target_model=target_net, tau=TAU)
            target_net.load_state_dict(policy_net.state_dict())
            print("target net weights updated")

        if np.min(rewards[-5:]) >= 200:
            break

    # save the rewards
    # rewards_path = 'training_rewards_{lr}_{eps_decay}_{network}.pkl'.format(lr=LR,eps_decay=EPS_DECAY,network='simple' )
    rewards_path = 'demo_training_rewards.pkl'
    save_rewards(rewards=rewards, path=rewards_path,option='training_rewards')

    # save the policy net
    # model_path = 'model_{lr}_{eps_decay}_{network}.pt'.format(lr=LR,eps_decay=EPS_DECAY,network='simple' )
    model_path = 'demo_model.pt'
    save_model(model=policy_net,path=model_path)