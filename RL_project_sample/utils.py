from collections import namedtuple
import torch
import torch.nn.functional as F
import random
import math
import os
import torch.nn as nn
import pickle

# global constants
N_STEPS_TIMEOUT = 20000
TARGET_UPDATE = 10
MEMORY_CAPACITY = 10000
BATCH_SIZE = 64
EPS_START = 0.9
EPS_END = 0.1


class DQN(nn.Module):
    def __init__(self,
                 state_size,
                 action_size,
                 seed=None,
                 fc1_unit=16,
                 fc2_unit=8):
        super(DQN, self).__init__()
        if seed is not None:
            self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_unit)
        self.fc2 = nn.Linear(fc1_unit, fc2_unit)
        self.fc3 = nn.Linear(fc2_unit, action_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


def select_action(state,
                  policy_net,
                  n_actions,
                  steps_done,
                  device,
                  eps_end,
                  eps_start,
                  eps_decay):
    sample = random.random()
    eps_threshold = eps_end + (eps_start - eps_end) * \
                    math.exp(-1. * steps_done / eps_decay)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            return torch.argmax(policy_net(state), dim=1).view(1, 1), steps_done
    else:
        return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long), steps_done


Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


# optimize the model for each steps
# Optimize the model
def optimize_model(memory,
                   batch_size,
                   device,
                   policy_net,
                   target_net,
                   optimizer,
                   gamma):
    if len(memory) < batch_size:
        return
    transitions = memory.sample(batch_size)
    batch = Transition(*zip(*transitions))

    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                       if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    state_action_values = policy_net(state_batch).gather(1, action_batch)
    next_state_values = torch.zeros(batch_size, device=device)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()

    expected_state_action_values = (next_state_values * gamma) + reward_batch

    # Compute Huber loss
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


# soft update the weights
def soft_update(local_model, target_model, tau):
    """Soft update model parameters.
    θ_target = τ*θ_local + (1 - τ)*θ_target
    Params
    =======
        local model (PyTorch model): weights will be copied from
        target model (PyTorch model): weights will be copied to
        tau (float): interpolation parameter
    """
    for target_param, local_param in zip(target_model.parameters(),
                                         local_model.parameters()):
        target_param.data.copy_(tau * local_param.data + (1 - tau) * target_param.data)


def save_rewards(rewards, path, option):
    # rewards_path = 'training_rewards_{lr}_{eps_decay}_{network}.pkl'.format(lr=0.001,eps_decay=10000,network='simple' )
    with open(os.path.join(option, path), 'wb') as f:
        pickle.dump(rewards, f)


def load_rewards(path, option):
    with open(os.path.join(option, path), 'rb') as f:
        saved_rewards = pickle.load(f)
    return saved_rewards


def save_model(model, path):
    torch.save(model.state_dict(), os.path.join('models', path))


def load_model(path, state_size, action_size, fc1_unit, fc2_unit):
    # 'model_{lr}_{eps_decay}_{network}.pt'.format(lr=0.001,eps_decay=10000,network='simple' )
    policy_net = DQN(state_size=state_size,
                     action_size=action_size,
                     fc1_unit=fc1_unit,
                     fc2_unit=fc2_unit)
    policy_net.load_state_dict(torch.load(os.path.join('models', path)))
    return policy_net
