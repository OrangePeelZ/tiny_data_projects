import matplotlib.pyplot as plt
import pickle
import os

def load_rewards(path, option):
    with open(os.path.join(option, path), 'rb') as f:
        saved_rewards = pickle.load(f)
    return saved_rewards

def plot_error_data(data, ax, algo_name, ylim=None):
    ax.plot([i[0] for i in data], [i[1] for i in data])
    ax.set_xlabel('number of simulations')
    ax.set_ylabel('ERROR')
    ax.set_title(algo_name)
    ax.ticklabel_format(axis='x',style='sci',scilimits=(0,0))
    if ylim:
        ax.set_ylim(ylim[0], ylim[1])


if __name__ == "__main__":
    ce_q_error = load_rewards(path='ce_q_error.pkl', option='data_store')
    foe_q_error = load_rewards(path='foe_q_error.pkl', option='data_store')
    friend_q_error = load_rewards(path='friend_q_error.pkl', option='data_store')
    q_learning_error = load_rewards(path='q_learning_error.pkl', option='data_store')

    fig, ax = plt.subplots(2, 2, figsize=(16, 10), dpi=80, facecolor='w', edgecolor='k')
    fig.suptitle('Error Convergence for Multi-agent Q Learning Algorithms')

    plot_error_data(data=q_learning_error, ax=ax[0][0], algo_name='Q Learning', ylim=(0,0.2))
    plot_error_data(data=friend_q_error, ax=ax[0][1], algo_name='Friend-Q', ylim=None)
    plot_error_data(data=foe_q_error, ax=ax[1][0], algo_name='Foe-Q', ylim=None)
    plot_error_data(data=ce_q_error, ax=ax[1][1], algo_name='uCE-Q', ylim=None)
    plt.savefig('plots/compare_algorithms.png')
    plt.close()