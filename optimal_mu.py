import pickle
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
def get_optimal_mu(size, good_acts, i, j, x, y, T):
    gridworld = GridWorld(good_acts, size)
    optimal_policy = policy_opt(i, j)

    total_reward = 0.0
    t = 1.0

    while t < T * 1.0:
        x, y, r = gridworld.take_action(x, y, optimal_policy[x][y])
        new_total_reward = (total_reward * (t - 1) + r) / t
        total_reward = new_total_reward
        t += 1.0

    # total reward is optimal mu
    return total_reward
