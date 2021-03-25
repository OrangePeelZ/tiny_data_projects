# Implement the generic multi-agent Q learning Framework
from soccer_env import SoccerGame
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os


def save_data(data, path, option):
    with open(os.path.join(option, path), 'wb') as f:
        pickle.dump(data, f)


def choose_action(action_space):
    return np.random.randint(action_space)

def choose_action_Q_learning(s, Q, eps, env):
    if np.random.random() < eps:
        # Case where we explore
        action = np.random.randint(env.nA)
    else:
        # Case where we exploit using Q-table
        [a1, a2] = np.argmax(Q[:, s, :], axis=1)
        print (a1, a2)
        action = env.action_vector_to_action_repr(a1, a2)
    return action

def multi_agent_friend_q(game_env,
                                gamma,
                                alpha,
                                alpha_decay,
                                n_simulation_steps,
                                epsilon,
                                epsilon_decay,
                                validate_pos=[2, 1, 1],
                                validate_direction=[2, 0],
                                seed=None):
    [min_alpha, max_alpha, alpha_decay_rate] = alpha_decay
    [min_epsilon, max_epsilon, epsilon_decay_rate] = epsilon_decay
    n_players = 2
    n_action_space = game_env.nA
    n_state_space = game_env.nS
    np.random.seed(seed)
    game_env.seed(seed)
    # initialize the states, actions and Q values
    Q = np.zeros((n_players, n_state_space, n_action_space))
    V = np.zeros((n_players, n_state_space))
    a = np.random.randint(n_action_space)

    # process validation point : A = S, B= Stick at inital state point
    validate_s = game_env.position_to_state(p_a=validate_pos[0], p_b=validate_pos[1], ball_pos=validate_pos[2])
    validate_a = game_env.action_vector_to_action_repr(a_a=validate_direction[0], a_b=validate_direction[1])
    print("validatation points are: position - {pos}; state - {st}; direction - {dir}; action repre - {ar}".format(
        pos=validate_pos,
        st=validate_s,
        dir=validate_direction,
        ar=validate_a))
    error_list = []
    ts_step = 0
    episode = 0

    while ts_step < n_simulation_steps:
        episode += 1
        s = game_env.reset()
        while True:
            Q_old = np.copy(Q)
            ts_step += 1
            s_next, r, d, _ = game_env.step(a)
            rs = [r, -r]
            for i in range(n_players):
                V[i, s_next] = np.max(Q[i, s_next, :])
                Q[i, s, a] = (1 - alpha) * Q[i, s, a] + alpha * ((1 - gamma) * rs[i] + gamma * V[i, s_next])
            a = choose_action(action_space=n_action_space)
            s = s_next

            error = np.abs(Q[0, validate_s, validate_a] - Q_old[0, validate_s, validate_a])
            if error > 0:
                error_list.append((ts_step, error))
            alpha = min_alpha + (max_alpha - min_alpha) * np.exp(-alpha_decay_rate * ts_step)
            # epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-epsilon_decay_rate * ts_step)

            if d:
                break
        if episode % 100 == 1:
            print(
                "In episode = {ep}, ts = {ts}: epsilon = {eps}, alpha = {alpha}".format(ep=episode,
                                                                                        eps=epsilon,
                                                                                        ts=ts_step,
                                                                                        alpha=alpha))

    return Q, V, error_list


if __name__ == '__main__':
    soccer = SoccerGame()
    n_simulation_steps = 1000000
    Q, V, error_list = multi_agent_friend_q(game_env=soccer,
                                            gamma=0.9,
                                            alpha=0.2,
                                            alpha_decay=[0.001, 0.2, 0.001], #0.001
                                            n_simulation_steps=n_simulation_steps,
                                            epsilon=1,
                                            epsilon_decay=[0.001, 1, 0.001], #[0.001, 1, 0.01]
                                            seed=0)
    save_data(data=error_list, path='friend_q_error.pkl',option='data_store')
    plt.plot([i[0] for i in error_list], [i[1] for i in error_list])
    plt.savefig('plots/friend_q_learning.png')
    how_to_play = np.argmax(Q[:,35,:], axis=1)
    print ( "based on Q_2, player B play: {0}".format(soccer.action_repr_to_action_vector(how_to_play[1])[1]))
