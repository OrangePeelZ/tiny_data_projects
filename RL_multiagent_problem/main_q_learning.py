# Implement the generic multi-agent Q learning Framework
from soccer_env import SoccerGame
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os


def save_data(data, path, option):
    with open(os.path.join(option, path), 'wb') as f:
        pickle.dump(data, f)


# for Q learning part.
def choose_action_Q_learning(s, Q, eps, per_person_action_space):
    if np.random.random() < eps:
        # Case where we explore
        a = np.random.randint(per_person_action_space)
    else:
        # Case where we exploit using Q-table
        a = np.argmax(Q[s, :])
    return a


def code_play_states(p, ball_pos):
    return 2 * p + ball_pos


def decode_play_states(s):
    return [int(s / 2), s % 2]


def decouple_states(s, env):
    p_a, p_b, ball_pos = env.state_to_position(s)
    return [code_play_states(p_a, ball_pos), code_play_states(p_b, ball_pos)]


def combine_states(s_a, s_b, env):
    p_a, ball_pos_a = decode_play_states(s_a)
    p_b, ball_pos_b = decode_play_states(s_b)
    if ball_pos_a != ball_pos_b:
        raise ValueError("something wrong with Environment!")
    return env.position_to_state(p_a, p_b, ball_pos_a)


# modified way for Q learning in the paper, which decays the learning rate
def multi_Q_learning(game_env,
                     gamma,
                     alpha,
                     alpha_decay,
                     n_simulation_steps,
                     epsilon,
                     epsilon_decay,
                     validate_s=35,
                     validate_a=2,
                     seed=None):
    [min_alpha, max_alpha, alpha_decay_rate] = alpha_decay
    [min_epsilon, max_epsilon, epsilon_decay_rate] = epsilon_decay
    n_action_space = game_env.nA
    per_user_action_space = int(np.sqrt(n_action_space))
    n_state_space = game_env.nS
    np.random.seed(seed)
    game_env.seed(seed)
    # initialize the states, actions and Q values
    Q1 = np.zeros((n_state_space, per_user_action_space))
    Q2 = np.zeros((n_state_space, per_user_action_space))
    V1 = np.zeros(n_state_space)
    V2 = np.zeros(n_state_space)
    action1 = np.random.randint(per_user_action_space)
    action2 = np.random.randint(per_user_action_space)

    error_list = []
    ts_step = 0
    episode = 0

    while ts_step < n_simulation_steps:
        episode += 1
        s = game_env.reset()
        while True:
            Q1_old = np.copy(Q1)
            ts_step += 1
            a = game_env.action_vector_to_action_repr(action1, action2)
            s_next, r, d, _ = game_env.step(a)
            [r1, r2] = [r, -r]
            # for player1/ playerA
            V1[s_next] = np.max(Q1[s_next, :])
            Q1[s, action1] = (1 - alpha) * Q1[s, action1] + alpha * ((1 - gamma) * r1 + gamma * V1[s_next])

            V2[s_next] = np.max(Q2[s_next, :])
            Q2[s, action2] = (1 - alpha) * Q2[s, action2] + alpha * ((1 - gamma) * r2 + gamma * V2[s_next])

            action1 = choose_action_Q_learning(s=s_next, Q=Q1, eps=epsilon,
                                               per_person_action_space=per_user_action_space)
            action2 = choose_action_Q_learning(s=s_next, Q=Q2, eps=epsilon,
                                               per_person_action_space=per_user_action_space)
            s = s_next

            error = np.abs(Q1[validate_s, validate_a] - Q1_old[validate_s, validate_a])
            if error > 0:
                error_list.append((ts_step, error))
            alpha = min_alpha + (max_alpha - min_alpha) * np.exp(-alpha_decay_rate * ts_step)
            epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-epsilon_decay_rate * ts_step)
            # print(s_decoupled[0], actions[0], Q[0, validate_s, validate_a])
            if d:
                break
        if episode % 100 == 1:
            print(
                "In episode = {ep}, ts = {ts}: epsilon = {eps}, alpha = {alpha}".format(ep=episode,
                                                                                        eps=epsilon,
                                                                                        ts=ts_step,
                                                                                        alpha=alpha))

    return Q1, Q2, V1, V2, error_list


if __name__ == '__main__':
    soccer = SoccerGame()
    n_simulation_steps = 5000000
    Q1, Q2, V1, V2, error_list = multi_Q_learning(game_env=soccer,
                                                  gamma=0.9,
                                                  alpha=0.2,
                                                  alpha_decay=[0.001, 0.2, 0.00001],
                                                  n_simulation_steps=n_simulation_steps,
                                                  epsilon=1,
                                                  epsilon_decay=[0.001, 1, 0.001], #[0.001, 1, 0.01] very good
                                                  seed=0)

    save_data(data=error_list, path='q_learning_error.pkl', option='data_store')

    plt.plot([i[0] for i in error_list], [i[1] for i in error_list])
    plt.savefig('plots/q_learning.png')
