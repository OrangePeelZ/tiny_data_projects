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


def choose_action(action_space):
    return np.random.randint(action_space)


def update_Q_function(Q_i, V_i, a, s, s_next, alpha, gamma, reward):
    return (1 - alpha) * Q_i[s, a] + alpha * ((1 - gamma) * reward + gamma * V_i[s_next])


def categorical_sample(prob_n):
    """
    Sample from categorical distribution
    Each row specifies class probabilities
    """
    prob_n = np.asarray(prob_n)
    csprob_n = np.cumsum(prob_n)
    return (csprob_n > np.random.rand()).argmax()


def transform_Q_function(Q, s, n_action_space):
    # Q[i, j] means the Q value on states when a1 take action i and a2 take action j
    # i and j are in 0,1,2,3,4,5
    size = int(np.sqrt(n_action_space))
    return Q[s, :].reshape(size, size).T


def solve_max_min(Q, size):
    equation_constrains = np.vstack((np.ones(size), -np.ones(size)))
    z_constrain_coef = np.hstack((np.zeros(2), np.ones(size))).reshape(-1, 1)
    constrain_coef = np.hstack((z_constrain_coef, np.vstack((equation_constrains, -Q)))).astype(np.double)
    # make all prob larger than 0
    pi_constr = np.hstack((np.zeros(size).reshape(-1, 1), -np.eye(size)))
    G = np.vstack((constrain_coef, pi_constr))
    c = -np.hstack((np.ones(1), np.zeros(size))).astype(np.double)
    b = np.hstack((np.array([1, -1]), np.zeros(2 * size))).astype(np.double)
    sol = solvers.lp(c=matrix(c),
                     G=matrix(G),
                     h=matrix(b),
                     solver='glpk',
                     options={'glpk': {'msg_lev': 'GLP_MSG_OFF'}})
    return sol


def foe_Q(Q, s, env):
    # format the LP problem
    # [Q(s,a1 = 0, a2 = 0), , Q(s, a=1, a2 = 0),Q(s, a=2, a2 = 0), Q(s, a=3, a2 = 0), Q(s, a=4, a2 = 0) ]
    size = int(np.sqrt(env.nA))
    Q_reformat = transform_Q_function(Q=Q, s=s, n_action_space=env.nA)
    return solve_max_min(Q_reformat, size=size)


def multi_Q_learning_foe_q(game_env,
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

    a = np.random.randint(n_action_space)
    Q = np.random.random((n_state_space, n_action_space))
    V = np.zeros((n_state_space))
    # initial equal probability for each action
    pi = np.ones((n_state_space, per_user_action_space)) * 1 / per_user_action_space
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
            solve_lp = equ_function(Q, s=s_next, env=game_env)
            V[s_next] = np.array(solve_lp['x'])[0]
            pi[s_next] = np.array(solve_lp['x'])[1:].reshape(5)
            Q[s, a] = update_Q_function(Q_i=Q, a=a, V_i=V, s=s, s_next=s_next, alpha=alpha, gamma=gamma, reward=r)
            a_prime = choose_action(action_space=n_action_space)
            s = s_next
            a = a_prime
            error = np.abs(np.abs(Q[validate_s][validate_a] - Q_old[validate_s][validate_a]))
            if error > 0:
                error_list.append((step, error))
            alpha = min_alpha + (max_alpha - min_alpha) * np.exp(-alpha_decay_rate * step)
            if d:
                break
        if episode % 100 == 1:
            print("In episode = {ep}, ts = {ts}: epsilon = {eps}, alpha = {alpha}, validate point action={at}". \
                  format(ep=episode,
                         eps=1,
                         alpha=alpha,
                         ts=step,
                         at=pi[validate_s]))
    return Q, V, pi, error_list


# if __name__ == '__main__':
soccer = SoccerGame()
Q, V, pi, error_list = multi_Q_learning_foe_q(game_env=soccer,
                                              equ_function=foe_Q,
                                              gamma=0.9,
                                              alpha=0.2,
                                              alpha_decay=[0.001, 0.2, 0.00001],
                                              n_simulation_steps=1000000,
                                              seed=0)

save_data(data=error_list, path='foe_q_error.pkl', option='data_store')
save_data(data=pi, path='foe_q_sigma.pkl', option='data_store')

plt.plot([i[0] for i in error_list], [i[1] for i in error_list])
plt.savefig('plots/foe_q_learning.png')
# [-4.96065194e-16  1.52837418e-17  4.60971256e-01  0.00000000e+00 5.39028744e-01]
