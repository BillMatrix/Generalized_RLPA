from environment.grid_world import GridWorld
import numpy as np
from optimal_mu import get_optimal_mu
from utils import span, complex_bound
from offpolicy_actor_critic.actor_critic import OffpolicyActorCritic
from importance_sampling.importance_sampling import importance_sampling
import copy


''' General RLPA Algorithm
    args:
        policy_lib: policy library contains all policy advice
        delta: confidence
        size, good_acts: parameters for gridworld
        T: time horizon
        method: off-policy policy gradient algorithm selection
        inter: boolean, whether or not the policy found by policy gradient
                needs to be within the coverage of the policy library
        '''


def general_rlpa(
    policy_lib, delta, size, good_acts, T, ind_act, ind_size, method, inter
):
    total_reward = 0

    init_x = np.random.random_integers(0, size - 1, 1)
    init_y = np.random.random_integers(0, size - 1, 1)

    x = int(init_x[0])
    y = int(init_y[0])

    optimal_mu = get_optimal_mu(size, good_acts, ind_act, ind_size, x, y, T)
    regret = []
    t = 1
    i = 0

    # Determine the full coverage of actions by the policy library
    # action_coverage = [[set() for _ in range(size)] for _ in range(size)]
    #
    # for policy in policy_lib.items():
    #     for i in range(size):
    #         for j in range(size):
    #             action_coverage[i][j].add(policy[i][j])

    n = {}
    mu_hat = {}
    R = {}
    K = {}

    for key, _ in policy_lib.items():
        # transform policy into stochastic policies
        for i in range(size):
            for j in range(size):
                action = policy_lib[key][i][j]
                policy_lib[key][i][j] = [0.0 for _ in range(4)]
                policy_lib[key][i][j][action - 1] = 1.0
        n[key] = 1.0
        mu_hat[key] = 0.0
        R[key] = 0.0
        K[key] = 1.0

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

            memory = []

            while t_i <= T_i and t <= T and v[m_B] <= n[m_B] \
                and mu_hat[m_B] - R[m_B] / (n[m_B] + v[m_B]) <= c[m_B] \
                    + complex_bound(H_hat, t, delta, n[m_B], 0, K[m_B]):
                t_i += 1

                policy = policy_lib[m_B]

                # sample stochastic policy
                move_direction = np.random.choice(4, 1, p=policy[x][y])[0] + 1

                x_new, y_new, r = gridworld.take_action(x, y, move_direction)

                memory += [(x, y, move_direction, r)]

                x = x_new
                y = y_new

                v[max_B_key] += 1.0
                R[max_B_key] += r
                total_reward += r

                regret += [optimal_mu - float(total_reward) / float(t)]

                t += 1

            K[m_B] += 1

            # if mu_hat[m_B] - R[m_B] / (n[m_B] + v[m_B]) > c[m_B] \
            #         + complex_bound(H_hat, t, delta, n[m_B], v[m_B], K[m_B]):
            #     policy_lib.pop(m_B, None)
            #     continue

            n[m_B] += v[m_B]

            mu_hat[m_B] = R[m_B] / n[m_B]

            if t_i < 100:
                continue

            # Derive new policy here
            new_policy = []
            if method == 'offpol_a3c':
                # use off-policy actor critic algorithm
                offpol_a3c = OffpolicyActorCritic(size)
                new_policy = offpol_a3c.derive_new_policy(
                    memory,
                    policy_lib[m_B]
                )

            # if inter:
            #     for i in range(size):
            #         for j in range(size):
            #             if not (new_policy[i][j] in action_coverage[i][j]):
            #                 continue

            exp_val, stdev = importance_sampling(
                new_policy,
                memory,
                policy_lib[m_B]
            )
            # if the lower bound of new policy is higher than the higher
            # bound of the cur policy, replace the old with new
            if exp_val > mu_hat[m_B]:
                print('new_policy_found')
                policy_lib[m_B] = copy.deepcopy(new_policy)
                mu_hat[m_B] = exp_val
                B[m_B] = exp_val + 1.96 * stdev

    return regret
