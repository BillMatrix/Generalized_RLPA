import numpy as np
from utils import state_feature, state_action_feature
import math


class OffpolicyActorCritic():

    '''
    args:
        size: size of the gridworld
    '''
    def __init__(
        self,
        size,
        lr_v=0.01,
        lr_w=0.01,
        lr_u=0.01,
        lam=0.05,
    ):
        self.lr_v = lr_v
        self.lr_w = lr_w
        self.lr_u = lr_u
        self.size = size
        self.lam = lam

    def compute_action_prob(self, u, x, y, action):
        target_s_a_feature = np.array(
            state_action_feature(x, y, self.size, action)
        )
        nominator = math.exp(np.dot(u, target_s_a_feature))
        denominator = 0.0
        for i in range(1, 5):
            s_a_feature = np.array(state_action_feature(x, y, self.size, i))
            denominator += math.exp(np.dot(u, s_a_feature))
        return nominator / denominator

    def compute_policy_gradient(self, u, x, y, action):
        target_s_a_feature = np.array(
            state_action_feature(x, y, self.size, action)
        )
        minus_term = 0.0
        for i in range(1, 5):
            s_b_feature = np.array(state_action_feature(x, y, self.size, i))
            minus_term += self.compute_action_prob(
                u,
                x,
                y,
                i,
            ) * s_b_feature
        return target_s_a_feature - minus_term

    def derive_new_policy(self, data_samples, behavior_pol):
        e_v = np.zeros(2 * self.size)
        e_u = np.zeros(2 * self.size + 4)
        w = np.zeros(2 * self.size)
        v = np.random.uniform(-0.03, 0.03, 2 * self.size)
        u = np.random.uniform(-0.03, 0.03, 2 * self.size + 4)
        T = len(data_samples)
        for i in range(T - 1):
            cur_state = (data_samples[i][0], data_samples[i][1])
            next_state = (data_samples[i + 1][0], data_samples[i + 1][1])
            action = data_samples[i][2]
            r = data_samples[i][3]
            cur_state_feature = np.array(state_feature(
                cur_state[0],
                cur_state[1],
                self.size,
            ))
            next_state_feature = np.array(state_feature(
                next_state[0],
                next_state[1],
                self.size,
            ))

            delta = r + np.dot(v, next_state_feature) \
                - np.dot(v, cur_state_feature)

            rho = self.compute_action_prob(
                u,
                cur_state[0],
                cur_state[1],
                action
            ) / behavior_pol[cur_state[0]][cur_state[1]][action - 1]

            # Update Critic
            e_v = rho * (cur_state_feature + self.lam * e_v)

            v_last_term = (1 - self.lam) * np.dot(w, e_v) * cur_state_feature
            v += self.lr_v * (delta * e_v - v_last_term)

            w_last_term = np.dot(w, cur_state_feature) * cur_state_feature
            w += self.lr_w * (delta * e_v - w_last_term)

            # Update Actor
            e_u = rho * (self.compute_policy_gradient(
                u,
                cur_state[0],
                cur_state[1],
                action,
            ) + self.lam * e_u)

            u += self.lr_u * delta * e_u

        # Construct policy based on u
        policy = [
            [[0.0, 0.0, 0.0, 0.0] for _ in range(self.size)]
            for _ in range(self.size)
        ]

        for i in range(self.size):
            for j in range(self.size):
                for k in range(4):
                    policy[i][j][k] = self.compute_action_prob(u, i, j, k + 1)

        return policy
