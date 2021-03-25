import numpy as np
from utils import load_rewards
import matplotlib.pyplot as plt
from main import GAMMA, LR, NETWORK, EPS_DECAY
from scipy.ndimage.filters import gaussian_filter1d
import sys

SUCCESSFUL_TRAIL = 200


def plot_rewards(ax, model_path, color, reward_option='training_rewards', extra_label=''):
    rewards_vector = np.array([None] * 1000)
    ysmoothed_vector = np.array([None] * 1000)

    rewards = load_rewards(model_path, reward_option)
    ysmoothed = gaussian_filter1d(rewards, sigma=6)
    rewards_vector[:len(rewards)] = rewards
    ysmoothed_vector[:len(ysmoothed)] = ysmoothed
    ax.plot(range(1000), rewards_vector, alpha=0.2, color=color, linestyle='dashed',
            label='Rewards {extra_label}'.format(extra_label=extra_label))
    ax.plot(range(1000), ysmoothed_vector, color=color,
            label='Smoothed Rewards {extra_label}'.format(extra_label=extra_label))
    ax.set_xlabel('number of episodes')
    ax.set_ylabel('Cumulative Rewards')
    ax.set_ylim(-600, 300)
    ax.legend()


def plot_hyperparam(ax, param, fixed_param, reward_option='training_rewards', color_list=["blue", "orange", "green"],
                    set_title=True):
    param_list = {'gamma': GAMMA,
                  'lr': LR,
                  'network': NETWORK,
                  'eps_decay': EPS_DECAY}
    for p, color in zip(param_list.get(param), color_list):
        fixed_param.update({param: p})
        train_rewards_path = '{reward_option}_{lr}_{eps_decay}_{gamma}_{network}.pkl'.format(**fixed_param,
                                                                                             reward_option=reward_option)
        plot_rewards(ax, train_rewards_path, color=color, reward_option=reward_option,
                     extra_label='{param}={v}'.format(param=param, v=p))
        if set_title:
            ax.set_title('Effect of {param} with the Fixed of Other Parameters'.format(param=param))


def load_all_test_rewards(hyper_params):
    test_rewards_dict = {}
    for (eps_decay, gamma, lr, network) in hyper_params:
        test_rewards_path = 'test_rewards_{lr}_{eps_decay}_{gamma}_{network}.pkl'.format(lr=lr,
                                                                                         eps_decay=eps_decay,
                                                                                         gamma=gamma,
                                                                                         network=network)
        test_rewards = load_rewards(test_rewards_path, 'test_rewards')
        if network not in test_rewards_dict.keys():
            test_rewards_dict[network] = {'_'.join([str(lr), str(eps_decay), str(gamma)]): test_rewards}
        else:
            test_rewards_dict[network].update({'_'.join([str(lr), str(eps_decay), str(gamma)]): test_rewards})
    return test_rewards_dict


if __name__ == '__main__':
    test_option = sys.argv[1]

    if test_option == 'demo':
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), dpi=80, facecolor='w', edgecolor='k')
        fig.suptitle('A simple DQN agent for Lunar Lander')
        plot_rewards(ax1, 'demo_training_rewards.pkl', color='cornflowerblue', reward_option='training_rewards')
        plot_rewards(ax2, 'demo_test_rewards.pkl', color='cornflowerblue', reward_option='test_rewards')
        plt.savefig('plots/demo.png')
        plt.close()

    elif test_option == 'reproduce':
        # first plot: train a simple agent
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), dpi=80, facecolor='w', edgecolor='k')
        fig.suptitle('A simple DQN agent for Lunar Lander')
        plot_rewards(ax1, 'demo_training_rewards.pkl', color='cornflowerblue', reward_option='training_rewards')
        plot_rewards(ax2, 'demo_test_rewards.pkl', color='cornflowerblue', reward_option='test_rewards')
        plt.savefig('plots/demo.png')
        plt.close()

        # plot eps_decay
        fig, ax = plt.subplots(1, 1, figsize=(16, 8), dpi=80, facecolor='w', edgecolor='k')
        plot_hyperparam(ax,
                        param='eps_decay',
                        fixed_param={'lr': 0.01,
                                     'gamma': 0.999,
                                     'network': 'simple'},
                        reward_option='training_rewards',
                        color_list=["blue", "orange", "green"])
        plt.savefig('plots/hyperparam_eps_decay.png')
        plt.close()

        # plot gamma
        fig, ax = plt.subplots(1, 1, figsize=(16, 8), dpi=80, facecolor='w', edgecolor='k')
        plot_hyperparam(ax,
                        param='gamma',
                        fixed_param={'lr': 0.01,
                                     'eps_decay': 10000,
                                     'network': 'simple'},
                        reward_option='training_rewards',
                        color_list=["blue", "orange", "green"])
        plt.savefig('plots/hyperparam_gamma.png')
        plt.close()

        # plot lr
        fig, ax = plt.subplots(1, 1, figsize=(16, 8), dpi=80, facecolor='w', edgecolor='k')
        plot_hyperparam(ax,
                        param='lr',
                        fixed_param={'gamma': 0.999,
                                     'eps_decay': 10000,
                                     'network': 'simple'},
                        reward_option='training_rewards',
                        color_list=["blue", "orange", "green"])
        plt.savefig('plots/hyperparam_lr.png')
        plt.close()

        # plot the best parameter fitting the best network structure
        test_rewards_dict = load_all_test_rewards(
            hyper_params=[(eps_decay, gamma, lr, network) for eps_decay in EPS_DECAY
                          for gamma in GAMMA
                          for lr in LR
                          for network in NETWORK.keys()])

        agg_rewards = {network: [(k, (np.array(v) > SUCCESSFUL_TRAIL).mean()) for k, v in param_dict.items()]
                       for network, param_dict in test_rewards_dict.items()}

        select_max_rewards = {}
        for k, v in agg_rewards.items():
            ind = [avg_rewards for _, avg_rewards in v].index(max([avg_rewards for _, avg_rewards in v]))
            params = [float(i) for i in v[ind][0].split('_')]
            params.extend([k])
            select_max_rewards.update({k: params})

        print(select_max_rewards)
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 16), dpi=80, facecolor='w', edgecolor='k')
        color_list = ["blue", "orange", "green"]
        for (lr, eps_decay, gamma, network), color in zip(select_max_rewards.values(), color_list):
            eps_decay = int(eps_decay)
            train_rewards_path = 'training_rewards_{lr}_{eps_decay}_{gamma}_{network}.pkl'.format(lr=lr,
                                                                                                  eps_decay=eps_decay,
                                                                                                  gamma=gamma,
                                                                                                  network=network)
            plot_rewards(ax1, train_rewards_path, color=color, reward_option='training_rewards',
                         extra_label=r'network={v}'.format(v=network))

        for (lr, eps_decay, gamma, network), color in zip(select_max_rewards.values(), color_list):
            eps_decay = int(eps_decay)
            test_rewards_path = 'test_rewards_{lr}_{eps_decay}_{gamma}_{network}.pkl'.format(lr=lr,
                                                                                             eps_decay=eps_decay,
                                                                                             gamma=gamma,
                                                                                             network=network)
            plot_rewards(ax2, test_rewards_path, color=color, reward_option='test_rewards',
                         extra_label=r'network={v}'.format(v=network))

        ax1.set_title('Train-time Rewards')
        ax2.set_title('Test-time Rewards')
        fig.suptitle('Best Hyper-parameter for Different Model Structure')
        plt.savefig('plots/policy_net_structure.png')
        plt.close()

    else:
        raise ValueError('Option not avaiable!')
