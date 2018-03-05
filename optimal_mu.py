import pickle
import numpy as np
from environment.grid_world import GridWorld
import math

''' load optimal pi computed by policy iteration'''
def policy_opt(i, j):
    return pickle.load(
        open(
            './optimal_policies/mdp_{0}_size_{1}'.format(str(i), str(j)),
            'rb',
        )
    )

''' compute mu according to the optimal policy'''
def get_optimal_mu(size, good_acts, i, j):
    gridworld = GridWorld(good_acts, size)
    optimal_policy = policy_opt(i, j)

    x = int(np.random.random_integers(0, size - 1, 1)[0])
    y = int(np.random.random_integers(0, size - 1, 1)[0])

    delta = 100.0
    total_reward = 0.0
    t = 1.0

    while t < 100.0 or delta > 0.0001:
        print([x, y])
        print(optimal_policy[x][y])
        x, y, r = gridworld.take_action(x, y, optimal_policy[x][y])
        new_total_reward = (total_reward * (t - 1) + r) / t
        delta = math.fabs(new_total_reward - total_reward)
        total_reward = new_total_reward
        t += 1.0

    # total reward is optimal mu
    return total_reward
