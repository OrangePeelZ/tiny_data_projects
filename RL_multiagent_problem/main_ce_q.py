# Implement the generic multi-agent Q learning Framework
from soccer_env import SoccerGame
import numpy as np
import matplotlib.pyplot as plt
from cvxopt import matrix, solvers
import pickle
import os

def save_data(data, path, option):
    with open(os.path.join(option, path), 'wb') as f:
        pickle.dump(data, f)


# for Q learning part.
def choose_action(action_space):
    return np.random.randint(action_space)


def update_Q_function(Q_i, V_i, a, s, s_next, alpha, gamma, reward):
    return (1 - alpha) * Q_i[s, a] + alpha * ((1 - gamma) * reward + gamma * V_i[s_next])


def format_c_coef(Q_s):
    Q1 = Q_s[0]
    Q2 = Q_s[1]
    return Q1 + Q2


def format_basic_coef(env):
    n_action_space = env.nA
    sum_eq_1 = np.vstack((np.ones(n_action_space), -np.ones(n_action_space)))  # (2*25)
    prob_greater_then_0 = -np.eye(n_action_space)  # (25*25)
    return np.vstack((sum_eq_1, prob_greater_then_0))


def format_ce_coef(Q_s, env):
    n_action_space = env.nA
    per_player_action = int(np.sqrt(n_action_space))
    # it has 25 constrain for row player and another 25 for col players; 5*2 constrains will be all zeros
    # the 25 coef will pi_(0,0), pi(0,1) ... pi(4,3),pi(4,4)
    ce_coef = np.zeros((n_action_space * 2, n_action_space))  # (50, 25)
    # row player
    row_player = Q_s[0].reshape((per_player_action, per_player_action)) - Q_s[0].reshape(
        (per_player_action, 1, per_player_action))
    for i in range(5):
        ce_coef[(i * 5):(i + 1) * 5, (i * 5):(i + 1) * 5] = row_player[i]

    # handle col players
    col_player = Q_s[1].reshape((per_player_action, per_player_action)).T - \
                 Q_s[1].reshape((per_player_action, per_player_action)).T.reshape(
                     (per_player_action, 1, per_player_action))
    for i in range(5):
        ce_coef[(25 + i * 5):(30 + 5 * i), i::5] = col_player[i]

    return ce_coef


def format_b_coef(env):
    n_action_space = env.nA
    sum_eq_1 = np.array([1, -1])
    prob_greater_then_0 = np.zeros(n_action_space)
    ce_coef = np.zeros(n_action_space * 2)
    return np.hstack((sum_eq_1, prob_greater_then_0, ce_coef))


def ce_Q(Q, s, env, default_V, default_sigma):
    # format the LP problem
    # [Q(s,a1 = 0, a2 = 0), , Q(s, a=1, a2 = 0),Q(s, a=2, a2 = 0), Q(s, a=3, a2 = 0), Q(s, a=4, a2 = 0) ]
    n_action_space = env.nA
    c = format_c_coef(Q_s=Q[:, s, :]).astype(np.double)

    basic_constrain_coef = format_basic_coef(env=env)
    ce_constrain_coef = format_ce_coef(Q_s=Q[:, s, :], env=env)
    b = format_b_coef(env=env).astype(np.double)
    G = np.vstack((basic_constrain_coef, ce_constrain_coef)).astype(np.double)  # 25+2+50 = 77
    sol = solvers.lp(c=matrix(c),
                     G=matrix(G),
                     h=matrix(b),
                     solver='glpk',
                     options={'glpk': {'msg_lev': 'GLP_MSG_OFF'}}) # options={'show_progress': False}
    try:
        sigma = np.array(sol['x']).reshape(n_action_space)
        V_s = np.sum(Q[:, s, :] * sigma, axis=1)
    except ValueError:
        V_s = default_V
        sigma = default_sigma
    return V_s, sigma


def multi_Q_learning_ce_q(game_env,
                          equ_function,
                          gamma,
                          alpha,
                          alpha_decay,
                          n_simulation_steps,
                          validate_pos=[2, 1, 1],
                          validate_direction=[2, 0],
                          seed=None):
    [min_alpha, max_alpha, alpha_decay_rate] = alpha_decay
    n_action_space = game_env.nA
    per_user_action_space = int(np.sqrt(n_action_space))
    n_state_space = game_env.nS
    np.random.seed(seed)
    game_env.seed(seed)
    # initialize the states, actions and Q values
    n_players = 2
    a = np.random.randint(n_action_space)
    Q = np.zeros((n_players, n_state_space, n_action_space))
    V = np.zeros((n_players, n_state_space))
    # initial equal probability for each action
    pi = np.ones((n_state_space, n_action_space)) * 1 / n_action_space
    # process validation point : A = S, B= Stick at inital state point
    validate_s = game_env.position_to_state(p_a=validate_pos[0], p_b=validate_pos[1], ball_pos=validate_pos[2])
    validate_a = game_env.action_vector_to_action_repr(a_a=validate_direction[0], a_b=validate_direction[1])
    print("validatation points are: position - {pos}; state - {st}; direction - {dir}; action repre - {ar}".format(
        pos=validate_pos,
        st=validate_s,
        dir=validate_direction,
        ar=validate_a))

    error_list = []
    step = 0
    episode = 0

    while step < n_simulation_steps:
        episode += 1
        s = game_env.reset()

        while True:
            step += 1
            Q_old = np.copy(Q)
            s_next, r, d, _ = game_env.step(a)
            # solve for play1
            V[:, s_next], pi[s_next] = equ_function(Q,
                                                    s=s_next,
                                                    env=game_env,
                                                    default_sigma=np.ones(n_action_space) * 1 / n_action_space,
                                                    default_V=np.zeros(n_players))
            Q[0, s, a] = update_Q_function(Q_i=Q[0], a=a, V_i=V[0], s=s, s_next=s_next, alpha=alpha, gamma=gamma,
                                           reward=r)
            Q[1, s, a] = update_Q_function(Q_i=Q[1], a=a, V_i=V[1], s=s, s_next=s_next, alpha=alpha, gamma=gamma,
                                           reward=-r)

            a_prime = choose_action(action_space=n_action_space)
            s = s_next
            a = a_prime

            error = np.abs(np.abs(Q[0, validate_s, validate_a] - Q_old[0, validate_s, validate_a]))
            if error > 0:
                error_list.append((step, error))
            alpha = min_alpha + (max_alpha - min_alpha) * np.exp(-alpha_decay_rate * step)

            if d:
                break
        if episode%100 ==1:
            print("In episode = {ep}, ts = {ts}: epsilon = {eps}, alpha = {alpha}, validate point action={at}". \
                  format(ep=episode,
                         eps=1,
                         alpha=alpha,
                         ts=step,
                         at=np.sum(pi[validate_s].reshape(5, 5), axis=1))) # np.sum(pi[validate_s].reshape(5, 5), axis=1)
    return Q, V, pi, error_list


# if __name__ == '__main__':
soccer = SoccerGame()
Q, V, pi, error_list = multi_Q_learning_ce_q(game_env=soccer,
                                             equ_function=ce_Q,
                                             gamma=0.9,
                                             alpha=0.2,
                                             alpha_decay=[0.001, 0.2, 0.00001], # 0.00001 and 1,000,000 this is good
                                             n_simulation_steps=1000000, #5000000
                                             seed=0)
save_data(data=error_list, path='ce_q_error.pkl',option='data_store')
save_data(data=pi, path='ce_q_sigma.pkl',option='data_store')

plt.plot([i[0] for i in error_list], [i[1] for i in error_list])
plt.savefig('plots/ce_q_learning.png')
# converge to [0.55645336 0.         0.44354664 0.         0.        ]
