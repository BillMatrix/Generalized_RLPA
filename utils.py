import numpy as np
import math


# function move decides which direction to move
# args:
#   probs: probability given a taken action
# rets:
#   integer in {0, 1, 2, 3, 4} indicating which direction to take
#   0 stay, 1 up, 2 down, 3 left, 4 right
def move(probs):
    return np.random.choice(5, 1, p=probs)[0]


# function to assign probability to actions according to good_action set
# args:
#   action: what action the agent wants to take in {1, 2, 3, 4}
#   1 up, 2 down, 3 left, 4 right
#   good_acts: good actions set
# rets:
#   a vector of probabilities of length 5 with index 1 to 4 corresponding to
#   the actions and 0 corresponding to not moving
def assign_probs(action, good_acts):
    probs = [0.0 for i in range(5)]
    if action in good_acts:
        for i in range(1, 5):
            if i == action:
                probs[i] = 0.85
            else:
                probs[i] = 0.05
    else:
        probs[0] = 0.85
        for i in range(1, 5):
            probs[i] = 0.0375

    return probs


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


def span(T_i):
	return math.log(T_i)

def complex_bound(h_hat, t, conf, n, v, k):
	return (h_hat + 1) * math.sqrt(48 * math.log(2 * t / conf) / (n + v)) + h_hat * k / (n + v)
