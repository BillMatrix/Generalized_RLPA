from utils import transform_policy_from_deterministic_to_stochastic


def get_optimal_mu(mdp):
    policy = [[0 for _ in range(mdp.size)] for _ in range(mdp.size)]

    for i in range(mdp.size):
        for j in range(mdp.size):
            if i == 0:
                policy[i][j] = 2
            elif i <= mdp.size / 2 and j <= mdp.size / 2:
                policy[i][j] = 0
            elif i <= mdp.size / 2 and j > mdp.size / 2:
                policy[i][j] = 2
            elif i > mdp.size / 2 and j <= mdp.size / 2:
                policy[i][j] = 0
            else:
                policy[i][j] = 2

    x = mdp.size / 2
    y = mdp.size / 2
    avg_reward = 0.0
    new_avg_reward = -1000.0
    t = 0

    while abs(avg_reward - new_avg_reward) > 0.001:
        avg_reward = new_avg_reward
        state, r = mdp.take_action((x, y), policy[int(x)][int(y)])
        x = state[0]
        y = state[1]
        t += 1
        new_avg_reward = (avg_reward * t + r) / (t + 1)

    return avg_reward
