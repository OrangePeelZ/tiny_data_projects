import gym
import torch
from itertools import count
from utils import load_model, save_rewards, N_STEPS_TIMEOUT
import sys
# import some constants from main
from main import GAMMA, LR, NETWORK, EPS_DECAY
from multiprocessing import Pool

REWARD_HARD_CUT = -400


def act_as_policy(state,
                  policy_net,
                  steps_done):
    with torch.no_grad():
        # t.max(1) will return largest column value of each row.
        # second column on max result is index of where max element was
        # found, so we pick action with the larger expected reward.
        return torch.argmax(policy_net(state), dim=1).view(1, 1), steps_done


def test(n_test_episodes, policy_net, env, device, seed=0, render=False):
    steps_done = 0
    env.seed(seed)
    rewards = []
    for i_episode in range(n_test_episodes):
        cumulative_reward = 0
        state = env.reset()
        state = torch.tensor([state])
        for t in count():
            if (cumulative_reward < REWARD_HARD_CUT) | (t > N_STEPS_TIMEOUT):
                break
            action, steps_done = act_as_policy(state=state,
                                               policy_net=policy_net,
                                               steps_done=steps_done)

            state_next, reward, done, _ = env.step(action.item())
            if render:
                env.render()
            cumulative_reward = cumulative_reward + reward
            # convert it to tensor
            state_next = torch.tensor([state_next], device=device)
            state = state_next
            if done:
                break
        if render:
            print("Cumulative Reward for Episode {i_episode} is: {cumulative_reward}". \
                  format(i_episode=i_episode, cumulative_reward=cumulative_reward))
        rewards.append(cumulative_reward)
    return rewards


def test_parallel(eps_decay, gamma, lr, network, seed, n_test_episodes, render, device):
    id = 'LunarLander-v2'
    env = gym.make(id).unwrapped
    n_actions = env.action_space.n
    n_states = env.observation_space.shape[0]
    print('start in valuating the agent with parameters: {pr}'.format(pr=[eps_decay, gamma, lr, network]))
    model_path = 'model_{lr}_{eps_decay}_{gamma}_{network}.pt'.format(lr=lr, eps_decay=eps_decay, gamma=gamma,
                                                                      network=network)
    if network not in NETWORK.keys():
        raise ValueError('Network key not existed!')

    fc1_unit, fc2_unit = NETWORK.get(network)
    policy_net = load_model(path=model_path,
                            fc1_unit=fc1_unit,
                            fc2_unit=fc2_unit,
                            state_size=n_states,
                            action_size=n_actions)
    rewards = test(n_test_episodes, policy_net, env, device=device, seed=seed, render=render)

    rewards_path = 'test_rewards_{lr}_{eps_decay}_{gamma}_{network}.pkl'.format(lr=lr, eps_decay=eps_decay,
                                                                                gamma=gamma,
                                                                                network=network)
    save_rewards(rewards=rewards, path=rewards_path, option='test_rewards')


if __name__ == '__main__':
    test_option = sys.argv[1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed = 12580
    n_test_episodes = 100
    render = False

    if test_option == 'demo':
        env_id = 'LunarLander-v2'
        env = gym.make(env_id).unwrapped
        n_actions = env.action_space.n
        n_states = env.observation_space.shape[0]

        path = 'demo_model.pt'
        policy_net = load_model(path=path,
                                state_size=n_states,
                                action_size=n_actions,
                                fc1_unit=16,
                                fc2_unit=8)
        rewards = test(n_test_episodes, policy_net, env, device=device, seed=seed, render=render)
        print("Rewards list for {n} episode is: {r}".format(n=n_test_episodes, r=rewards))
        save_rewards(rewards=rewards, path='demo_test_rewards.pkl', option='test_rewards')

    elif test_option == 'hyperparam':

        hyper_params = [(eps_decay, gamma, lr, network, seed, n_test_episodes, render, device)
                        for eps_decay in EPS_DECAY
                        for gamma in GAMMA
                        for lr in LR
                        for network in NETWORK.keys()]

        pool = Pool(6)
        pool.starmap(test_parallel, hyper_params)
        pool.close()
    else:
        raise ValueError('Option not avaiable!')
