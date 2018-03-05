from environment.grid_world import GridWorld
import math
import numpy as np
from optimal_mu import get_optimal_mu


'''
Compute Span
    args:
        T_i: episode length
    rets:
        H_hat'''
def span(T_i):
    return math.log(T_i)


'''
Compute the complex bound
    args: H_hat, t, delta, n, v, k
    rets: bound '''
def complex_bound(H_hat, t, delta, n, v, k):
    first_term = math.sqrt(48.0 * math.log(2.0 * t / delta) / (n + v))
    second_term = H_hat * k / (n + v)
    return (H_hat + 1) * first_term + second_term


''' RLPA Algorithm
    Refer to arxiv.org/pdf/1305.1027.pdf for more details'''
def rlpa(policy_lib, delta, size, good_acts, T, index_act, index_size):
    total_reward = 0
    optimal_mu = get_optimal_mu(size, good_acts, index_act, index_size)
    regret = []
    t = 1
    i = 0

    n = {}
    mu_hat = {}
    R = {}
    K = {}

    for key, _ in policy_lib.items():
        n[key] = 1.0
        mu_hat[key] = 0.0
        R[key] = 0.0
        K[key] = 1.0

    init_x = np.random.random_integers(0, size - 1, 1)
    init_y = np.random.random_integers(0, size - 1, 1)

    x = int(init_x[0])
    y = int(init_y[0])

    while t <= T:
        t_i = 0
        T_i = 2 ** i
        H_hat = span(T_i)
        i += 1

        gridworld = GridWorld(good_acts, size)

        while t_i <= T_i and t <= T and len(policy_lib) != 0:
            print(t)
            c = {}
            B = {}
            v = {}
            max_B_key = -1
            max_B = -100.0

            for key, _ in policy_lib.items():
                c[key] = complex_bound(H_hat, t, delta, n[key], 0, K[key])
                B[key] = mu_hat[key] + c[key]
                if B[key] > max_B:
                    max_B = B[key]
                    max_B_key = key

            v[max_B_key] = 1.0

            m_B = max_B_key

            while t_i <= T_i and t <= T and v[m_B] <= n[m_B] \
                and mu_hat[m_B] - R[m_B] / (n[m_B] + v[m_B]) > c[m_B] \
                    + complex_bound(H_hat, t, delta, n[m_B], 0, K[m_B]):
                t_i += 1

                policy = policy_lib[m_B]

                x, y, r = gridworld.take_action(x, y, policy[x][y])

                v[max_B_key] += 1.0
                R[max_B_key] += r
                total_reward += r

                regret += [optimal_mu - float(total_reward) / float(t)]

                t += 1

            K[m_B] += 1
            if mu_hat[m_B] - R[m_B] / (n[m_B] + v[m_B]) > c[m_B] \
                    + complex_bound(H_hat, t, delta, n[m_B], v[m_B], K[m_B]):
                policy_lib.pop(m_B, None)

            n[m_B] += v[m_B]

            mu_hat[m_B] = R[m_B] / n[m_B]

    return regret
