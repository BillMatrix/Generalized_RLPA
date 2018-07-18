import numpy as np
import math


def converge(new_Q, Q, diff):
    for key, _ in new_Q.items():
        if np.linalg.norm(np.array(new_Q[key]) - np.array(Q[key])) > diff:
            return False

    return True


def transform_policy_from_deterministic_to_stochastic(size, policy_lib):
    new_policy_lib = {}
    for key, _ in policy_lib.items():
        new_policy_lib[key] = {}
        # transform policy into stochastic policies
        for i in range(size):
            for j in range(size):
                action = policy_lib[key][i][j]
                new_policy_lib[key][(i, j)] = [0.0 for _ in range(4)]
                new_policy_lib[key][(i, j)][action] = 1.0

    return new_policy_lib

# function move decides which direction to move
# args:
#   probs: probability given a taken action
# rets:
#   integer in {0, 1, 2, 3, 4} indicating which direction to take
#   4 stay, 0 up, 1 down, 2 left, 3 right
def move(probs):
    return np.random.choice(5, 1, p=probs)[0]


def select_action(probs):
    return np.random.choice(4, 1, p=probs)[0]

# function to extract features from a state from a gridworld
# args:
#   x, y: coordinates in a GridWorld
#   size: size of the gridworld
# rets:
#   concatenated onehot representation of (x, y)
def state_feature(x, y, size):
    x_feature = [0.0 for _ in range(size)]
    y_feature = [0.0 for _ in range(size)]
    x_feature[x] = 1.0
    y_feature[y] = 1.0
    return x_feature + y_feature


# function to extract features from a state action pair from a GridWorld
# args:
#   x, y: coordinates in a GridWorld
#   size: size of the GridWorld
#   a: one of 1, 2, 3, 4 representing up, down, left and right
# rets:
#   concatenated onehot representation of (x, y, a)
def state_action_feature(x, y, size, a):
    state_feat = state_feature(x, y, size)
    a_feature = [0.0 for _ in range(4)]
    a_feature[a-1] = 1.0
    return state_feat + a_feature


# KL Divergence of two policies
# args:
#   new_policy: derived policy
#   behavior_pol: behavior policy used so far
#   data_samples: history
def KL_Diverge(new_policy, behavior_pol, data_samples):
    p_new = 1.0
    p_new_over_p_beh = 1.0
    KL = 0.0
    T = len(data_samples)
    record = []
    for i in range(T):
        cur_state = (data_samples[i][0], data_samples[i][1])
        cur_action = data_samples[i][2]
        prob_beh_policy_cur_action \
            = behavior_pol[cur_state[0]][cur_state[1]][cur_action - 1]
        prob_new_policy_cur_action \
            = new_policy[cur_state[0]][cur_state[1]][cur_action - 1]
        p_new = p_new * prob_new_policy_cur_action
        record += [prob_new_policy_cur_action / prob_beh_policy_cur_action]
        p_new_over_p_beh = p_new_over_p_beh * \
            prob_new_policy_cur_action / prob_beh_policy_cur_action
        if i >= 5000:
            p_new_over_p_beh = p_new_over_p_beh / record[i - 5000]
        if p_new <= 0:
            KL -= 1
        else:
            KL += p_new * math.log(p_new_over_p_beh)

    return abs(KL) > T * 0.1


def span(T_i):
	return math.log(T_i)

def complex_bound(h_hat, t, conf, n, v, k):
	return (h_hat + 1) * math.sqrt(48 * math.log(2 * t / conf) / (n + v)) + h_hat * k / (n + v)
