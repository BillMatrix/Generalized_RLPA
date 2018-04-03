import numpy as np
from optimal_mu import get_optimal_mu
from environment.grid_world import GridWorld
import math

'''
    For details of this algorithm, please visit
    http://www.dcsc.tudelft.nl/~sc4081/assign/pap/gabriel_fernandez06_paper.pdf
'''
def policy_reuse(
    policy_lib,
    size,
    good_acts,
    ind_act,
    ind_size,
    K,
    H,
    tao=0,
    d_tao=5,
    phi=1,
    v=0.95,
    alpha=0.05,
):
    init_x = np.random.random_integers(0, size - 1, 1)
    init_y = np.random.random_integers(0, size - 1, 1)

    x = int(init_x[0])
    y = int(init_y[0])

    optimal_mu \
        = get_optimal_mu(size, good_acts, ind_act, ind_size, x, y, K * H)

    gridworld = GridWorld(good_acts, size)

    Q = [
        [[0.0 for _ in range(4)] for _ in range(size)]
        for _ in range(size)
    ]

    W = {
        100: 0.0,
    }

    U = {
        100: 0.0,
    }

    for key, _ in policy_lib.items():
        W[key] = 0.0
        U[key] = 0.0

    regrets = []

    for k in range(1, K + 1):
        print(k)
        policy_index = sample_policy(W, tao)

        if policy_index == 100:
            R, Q, cur_regrets, x, y \
                = standard_Q_learning(
                    Q,
                    H,
                    x,
                    y,
                    gridworld,
                    optimal_mu,
                    alpha,
            )
            regrets += cur_regrets
        else:
            R, Q, policy_lib[policy_index], cur_regrets, x, y \
                = pi_reuse(
                    policy_lib[policy_index],
                    Q,
                    H,
                    phi,
                    v,
                    x,
                    y,
                    gridworld,
                    optimal_mu,
                    size,
                    alpha,
            )
            regrets += cur_regrets

        W[key] = (W[key] * U[key] + R) / (U[key] + 1.0)
        U[key] += 1.0
        tao += d_tao

    return regrets


def pi_reuse(policy, Q, H, phi, v, x, y, grid_world, optimal_mu, size, alpha):
    regrets = []
    total_reward = 0.0
    t = 0.0
    Q = [
        [[0.0 for _ in range(4)] for _ in range(size)]
        for _ in range(size)
    ]
    for _ in range(H):
        probs = [phi, 1.0 - phi]
        past = np.random.choice(2, 1, p=probs)[0]
        a = 0

        if past == 0:
            a = policy[x][y]

        else:
            best_a = get_policy(Q, x, y)
            epsilon = 0.05
            p = [epsilon for _ in range(4)]
            p[best_a - 1] = 1 - 3 * epsilon
            a = np.random.choice(4, 1, p=p)[0]

        next_x, next_y, r = grid_world.take_action(x, y, a)
        total_reward += r
        t += 1.0

        next_a = get_policy(Q, next_x, next_y)
        Q[x][y][a - 1] = (1 - alpha) * Q[x][y][a - 1] \
            + alpha * (r + Q[next_x][next_y][next_a - 1])
        regrets += [optimal_mu - total_reward / t]

        x = next_x
        y = next_y

        phi = phi * v

    new_policy = [
        [0 for _ in range(size)]
        for _ in range(size)
    ]

    for i in range(size):
        for j in range(size):
            new_policy[i][j] = get_policy(Q, i, j)

    regrets = [sum(regrets) * 1.0 / len(regrets) for _ in range(len(regrets))]

    return total_reward / t, Q, new_policy, regrets, x, y


def standard_Q_learning(Q, H, x, y, grid_world, optimal_mu, alpha):
    regrets = []
    total_reward = 0.0
    t = 0.0
    for _ in range(H):
        cur_a = get_policy(Q, x, y)

        next_x, next_y, r = grid_world.take_action(x, y, cur_a)
        total_reward += r
        t += 1.0

        next_a = get_policy(Q, next_x, next_y)
        Q[x][y][cur_a - 1] = Q[x][y][cur_a - 1] + \
            alpha * (r + Q[next_x][next_y][next_a - 1] - Q[x][y][cur_a - 1])

        x = next_x
        y = next_y

        regrets += [optimal_mu - total_reward / t]

    regrets = [sum(regrets) * 1.0 / len(regrets) for _ in range(len(regrets))]

    return total_reward / t, Q, regrets, x, y


def get_policy(Q, x, y):
    scores = Q[x][y]
    best_action = 1
    for i in range(4):
        if scores[i] > scores[best_action]:
            best_action = i

    return best_action + 1


def sample_policy(W, tao):
    probs = [0.0 for _ in range(len(W))]
    keys = []
    for key, _ in W.items():
        keys += [key]

    denominator_softmax = 0.0
    for _, value in W.items():
        denominator_softmax += math.exp(tao * value)

    for i in range(len(probs)):
        probs[i] = math.exp(tao * W[keys[i]]) / denominator_softmax

    return keys[np.random.choice(len(probs), 1, p=probs)[0]]
