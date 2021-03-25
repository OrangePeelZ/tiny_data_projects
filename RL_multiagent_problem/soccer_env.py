# this script builds the soccer environment
# 1. If players try to move to the same cell from different positions the second player doesn't move and
# loses the ball (if in possession)
#
# 2. If a player sticks or tries to move to a boundary (i.e. stays at the same place) and the other player tries to
# move into his spot, the player already in that cell gets the ball (if applicable) and the other player doesn't move
# (independent of who moves first)
#
# 3. It's fine for players to switch positions and nobody loses the ball

import numpy as np
from gym.envs.toy_text import discrete


class SoccerGame(discrete.DiscreteEnv):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(self, nrol=2, ncol=4, **kwargs):
        self.nrol = nrol
        self.ncol = ncol
        self.max_row = nrol - 1
        self.max_col = ncol - 1
        # Define action and observation space
        N_ACTION_PER_PERSON = 5
        N_DISCRETE_ACTIONS = N_ACTION_PER_PERSON * N_ACTION_PER_PERSON
        # the two players can not occupied the same cell; either player will have the ball
        N_STATES_PER_PERSON = 8
        N_STATES_BALL = 2
        N_STATES = N_STATES_PER_PERSON * N_STATES_PER_PERSON * N_STATES_BALL

        # always initiate at coord_a=[0,2], coord_b = [0,1], ball_pos = 1
        initial_state_distrib = np.zeros(N_STATES)
        init_coord_a = [0, 2]
        init_coord_b = [0, 1]
        init_ball_pos = 1
        self.initial_state = self.position_to_state(p_a=self.coord_to_position(init_coord_a),
                                               p_b=self.coord_to_position(init_coord_b),
                                               ball_pos=init_ball_pos)
        initial_state_distrib[self.initial_state] = 1.0

        P = {state: {action: []
                     for action in range(N_DISCRETE_ACTIONS)} for state in range(N_STATES)}

        for p_a in range(8):
            for p_b in range(8):
                for ball_pos in range(2):
                    for a_r in range(25):
                        s = self.position_to_state(p_a=p_a, p_b=p_b, ball_pos=ball_pos)
                        [coord_a_x, coord_a_y], [coord_b_x, coord_b_y] = self.position_to_coord(
                            p_a), self.position_to_coord(p_b)
                        a_a, a_b = self.action_repr_to_action_vector(a_r)
                        [intent_coord_a_x, intent_coord_a_y], [intent_coord_b_x, intent_coord_b_y] = \
                            self.move_to_intent_place(coord_a_x=coord_a_x, coord_a_y=coord_a_y, coord_b_x=coord_b_x,
                                                      coord_b_y=coord_b_y, a_r=a_r)
                        if_move_a = True if a_a != 0 else False
                        if_move_b = True if a_b != 0 else False
                        # handle invalid intent
                        if self.check_if_hit_bounder(intent_x=intent_coord_a_x, intent_y=intent_coord_a_y):
                            [intent_coord_a_x, intent_coord_a_y] = [coord_a_x, coord_a_y]
                            if_move_a = False
                        if self.check_if_hit_bounder(intent_x=intent_coord_b_x, intent_y=intent_coord_b_y):
                            [intent_coord_b_x, intent_coord_b_y] = [coord_b_x, coord_b_y]
                            if_move_b = False

                        # handle collide a,b to move
                        if if_move_a & if_move_b & \
                                (intent_coord_a_x == intent_coord_b_x) & (intent_coord_a_y == intent_coord_b_y):

                            # a_move and b dont move
                            ball_pos_next, reward, done = self.handle_collide(if_move_a=True,
                                                                              intent_coord_a_y=intent_coord_a_y,
                                                                              intent_coord_b_y=intent_coord_b_y)
                            p_a_next = self.coord_to_position(coord=[intent_coord_a_x, intent_coord_a_y])
                            p_b_next = p_b
                            next_s = self.position_to_state(p_a=p_a_next, p_b=p_b_next, ball_pos=ball_pos_next)
                            P[s][a_r].append((0.5, next_s, reward, done))

                            # b move and a dont
                            ball_pos_next, reward, done = self.handle_collide(if_move_a=False,
                                                                              intent_coord_a_y=intent_coord_a_y,
                                                                              intent_coord_b_y=intent_coord_b_y)
                            p_a_next = p_a
                            p_b_next = self.coord_to_position(coord=[intent_coord_b_x, intent_coord_b_y])
                            next_s = self.position_to_state(p_a=p_a_next, p_b=p_b_next, ball_pos=ball_pos_next)
                            P[s][a_r].append((0.5, next_s, reward, done))

                        elif if_move_a & (not if_move_b) & \
                                (intent_coord_a_x == intent_coord_b_x) & (intent_coord_a_y == intent_coord_b_y):
                            # a move, b not move, a tries to move to b's position; b always get the ball and a stays
                            p_a_next, p_b_next, ball_pos_next = p_a, p_b, 1
                            # check if b wins
                            reward, done = self.check_done(ball_pos=ball_pos_next,
                                                           coord_a_y=coord_a_y,
                                                           coord_b_y=coord_b_y)
                            next_s = self.position_to_state(p_a=p_a_next, p_b=p_b_next, ball_pos=ball_pos_next)
                            P[s][a_r].append((1.0, next_s, reward, done))

                        elif (not if_move_a) & if_move_b & \
                                (intent_coord_a_x == intent_coord_b_x) & (intent_coord_a_y == intent_coord_b_y):
                            # b move, a not move, b tries to move to a's position; a always get the ball and b stays
                            p_a_next, p_b_next, ball_pos_next = p_a, p_b, 0
                            reward, done = self.check_done(ball_pos=ball_pos_next,
                                                           coord_a_y=coord_a_y,
                                                           coord_b_y=coord_b_y)
                            next_s = self.position_to_state(p_a=p_a_next, p_b=p_b_next, ball_pos=ball_pos_next)
                            P[s][a_r].append((1.0, next_s, reward, done))

                        else:
                            p_a_next = self.coord_to_position(coord=[intent_coord_a_x, intent_coord_a_y])
                            p_b_next = self.coord_to_position(coord=[intent_coord_b_x, intent_coord_b_y])
                            ball_pos_next = ball_pos

                            reward, done = self.check_done(ball_pos=ball_pos_next,
                                                           coord_a_y=intent_coord_a_y,
                                                           coord_b_y=intent_coord_b_y)
                            next_s = self.position_to_state(p_a=p_a_next, p_b=p_b_next, ball_pos=ball_pos_next)
                            P[s][a_r].append((1.0, next_s, reward, done))

        discrete.DiscreteEnv.__init__(
            self, N_STATES, N_DISCRETE_ACTIONS, P, initial_state_distrib)

    def position_to_coord(self, p):
        return [int(p / 4), p % 4]

    def coord_to_position(self, coord):
        x, y = coord
        return (x * 4 + y)

    def position_to_state(self, p_a, p_b, ball_pos):
        return 8 * 2 * p_a + 2 * p_b + ball_pos

    def state_to_position(self, s):
        ball_pos = s % 2
        p_b = int(s / 2) % 8
        p_a = int(int(s / 2) / 8) % 8
        return p_a, p_b, ball_pos

    def decode_action(self, a):
        # a is in {0:stand, 1:E, 2:S, 3:W, 4:N}
        action_dict = {0: [0, 0],
                       1: [0, 1],
                       2: [1, 0],
                       3: [0, -1],
                       4: [-1, 0]}
        try:
            return action_dict[a]
        except:
            raise ValueError('invalid action!')

    def action_repr_to_action_vector(self, a_r):
        # each player has 5 actions and there are 25 actions in total, a_r =0, 1, ..., 24
        # a_r = 1 means a_a = 0 and a_b = 1; a = 24 means a_a = 4 and a_b = 4
        return int(a_r / 5), a_r % 5

    def encode_actions(self, move):
        move_x, move_y = move
        if (move_x == 0) & (move_y == 0):
            return 0
        elif (move_x == 0) & (move_y == 1):
            return 1
        elif (move_x == 1) & (move_y == 0):
            return 2
        elif (move_x == 0) & (move_y == -1):
            return 3
        elif (move_x == -1) & (move_y == 0):
            return 4
        else:
            raise ValueError('invalid move')

    def action_vector_to_action_repr(self, a_a, a_b):
        return a_a * 5 + a_b

    def move_to_intent_place(self, coord_a_x, coord_a_y, coord_b_x, coord_b_y, a_r):
        a_a, a_b = self.action_repr_to_action_vector(a_r)
        [move_a_x, move_a_y], [move_b_x, move_b_y] = self.decode_action(a_a), self.decode_action(a_b)
        intent_coord_a_x, intent_coord_a_y = coord_a_x + move_a_x, coord_a_y + move_a_y
        intent_coord_b_x, intent_coord_b_y = coord_b_x + move_b_x, coord_b_y + move_b_y
        return [intent_coord_a_x, intent_coord_a_y], [intent_coord_b_x, intent_coord_b_y]

    def check_if_hit_bounder(self, intent_x, intent_y):
        if (intent_x > self.max_row) | (intent_x < 0) | (intent_y > self.max_col) | (intent_y < 0):
            return True
        else:
            return False


    def check_done(self,ball_pos, coord_a_y, coord_b_y, verbose=False):
        if (coord_a_y == 0) & (ball_pos == 0):
            reward, done = 100, True
            if verbose:
                print("player a score it! player a is {r}".format(r=reward))
            return reward, done
        elif (coord_b_y == 0) & (ball_pos == 1):
            reward, done = 100, True
            if verbose:
                print("player b score it! player a is {r}".format(r=reward))
            return reward, done

        elif (coord_a_y == self.max_col) & (ball_pos == 0):
            reward, done = -100, True
            if verbose:
                print("player a score it! player a is {r}".format(r=reward))
            return reward, done

        elif (coord_b_y == self.max_col) & (ball_pos == 1):
            reward, done = -100, True
            if verbose:
                print("player b score it! player a is {r}".format(r=reward))
            return reward, done
        else:
            return 0, False

    def handle_collide(self, if_move_a, intent_coord_a_y, intent_coord_b_y):
        # (1.0, new_state, reward, done)
        # call the function when both a and b move and cause a collide
        if if_move_a:
            ball_pos_next = 0
            reward, done = self.check_done(ball_pos=ball_pos_next,
                                           coord_a_y=intent_coord_a_y,
                                           coord_b_y=intent_coord_b_y)
        else:
            # when the player b has the right to move； B will always have the ball possessed
            ball_pos_next = 1
            reward, done = self.check_done(ball_pos=ball_pos_next,
                                           coord_a_y=intent_coord_a_y,
                                           coord_b_y=intent_coord_b_y)
        return ball_pos_next, reward, done

    def render(self, mode='human', close=False):
        p_a, p_b, ball_pos = self.state_to_position(self.s)
        player_a = 'A' if ball_pos == 0 else 'a'
        player_b = 'B' if ball_pos == 1 else 'b'
        coord_a = self.position_to_coord(p_a)
        coord_b = self.position_to_coord(p_b)
        str_mat_r0 = ['|', ' ', '|', ' ', '|', ' ', '|', ' ', '|']
        str_mat_r1 = ['|', ' ', '|', ' ', '|', ' ', '|', ' ', '|']
        if coord_a[0] == 0:
            if str_mat_r0[(coord_a[1] * 2 + 1)] != ' ':
                raise ValueError('Collision!')
            str_mat_r0[(coord_a[1] * 2 + 1)] = player_a
        else:
            if str_mat_r1[(coord_a[1] * 2 + 1)] != ' ':
                raise ValueError('Collision!')
            str_mat_r1[(coord_a[1] * 2 + 1)] = player_a

        if coord_b[0] == 0:
            if str_mat_r0[(coord_b[1] * 2 + 1)] != ' ':
                raise ValueError('Collision!')
            str_mat_r0[(coord_b[1] * 2 + 1)] = player_b
        else:
            if str_mat_r1[(coord_b[1] * 2 + 1)] != ' ':
                raise ValueError('Collision!')
            str_mat_r1[(coord_b[1] * 2 + 1)] = player_b
        print('-----------------')
        print(' '.join(str_mat_r0))
        print('-----------------')
        print(' '.join(str_mat_r1))
        print('-----------------')

    def step_and_render(self, a):
        # render
        a_a, a_b = self.action_repr_to_action_vector(a)
        coord_a, coord_b = self.decode_action(a_a), self.decode_action(a_b)
        s_next, r, d, prob = self.step(a)
        print("take action: {moves}, represent = {ar}, reward = {rw}".format(moves=[coord_a, coord_b], ar=a, rw=r))
        self.render()
        return s_next, r, d, prob


if __name__ == '__main__':
    env = SoccerGame()
    n_case = 1
    if n_case == 1:
        env.render()
        # b move right and A stays
        a_a = [0, 0]
        a_b = [0, 1]
        ar = env.action_vector_to_action_repr(env.encode_actions(a_a), env.encode_actions(a_b))
        env.step_and_render(a=ar)

        # b move right and A stays
        a_a = [0, 0]
        a_b = [1, 0]
        ar = env.action_vector_to_action_repr(env.encode_actions(a_a), env.encode_actions(a_b))
        env.step_and_render(a=ar)

        # b move right and A stays
        a_a = [0, 0]
        a_b = [1, 0]
        ar = env.action_vector_to_action_repr(env.encode_actions(a_a), env.encode_actions(a_b))
        env.step_and_render(a=ar)

        # try to move the same place
        a_a = [0, -1]
        a_b = [-1, 0]
        ar = env.action_vector_to_action_repr(env.encode_actions(a_a), env.encode_actions(a_b))
        env.step_and_render(a=ar)

        # a continue move to the goal zone
        a_a = [0, -1]
        a_b = [-1, 0]
        ar = env.action_vector_to_action_repr(env.encode_actions(a_a), env.encode_actions(a_b))
        env.step_and_render(a=ar)
    if n_case == 2:
        env.render()
        # a and b shift to the right
        a_a = [0, 1]
        a_b = [0, 1]
        ar = env.action_vector_to_action_repr(env.encode_actions(a_a), env.encode_actions(a_b))
        env.step_and_render(a=ar)

        # a and b shift to the right again
        a_a = [0, 1]
        a_b = [0, 1]
        ar = env.action_vector_to_action_repr(env.encode_actions(a_a), env.encode_actions(a_b))
        env.step_and_render(a=ar)
