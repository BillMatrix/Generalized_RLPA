import numpy as np


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
