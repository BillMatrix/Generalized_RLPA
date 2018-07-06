import pickle
import math

def get_optimal_mu(mdp):
    policy = policy_iteration(mdp)
    x = mdp.size / 2
    y = mdp.size / 2
    avg_reward = 0.0
    t = 0

    while abs(avg_reward - new_avg_reward) > 0.001:
        state, r = mdp.take_action(policy[x][y])
        x = state[0]
        y = state[1]
        t += 1
        avg_reward = (avg_reward * t + r) / (t + 1)

    return avg_reward
