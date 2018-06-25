from environment.grid_world import GridWorld
import numpy as np
from optimal_mu import get_optimal_mu
from utils import span, complex_bound
from model.offpolicy_actor_critic.actor_critic import OffpolicyActorCritic
from model.importance_sampling.importance_sampling import importance_sampling
from model.robust_dynamic_programming.robust_dp import RobustDP
import copy


def update_behavior_pol(behavior_pol, new_policy, t, size):
    if len(behavior_pol) == 0:
        behavior_pol = copy.deepcopy(new_policy)
    else:
        for i in range(size):
            for j in range(size):
                for k in range(4):
                    behavior_pol[i][j][k] = float(
                        t * behavior_pol[i][j][k] + new_policy[i][j][k]
                    ) / (t + 1)

    return behavior_pol

def initialize_params(policy_lib):
    transitions = {}
    core_transitions = {}
    rewards = {}
    core_rewards = {}

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
        transitions[key] = [
            [[[1 for _ in range(5)] for _ in range(4)] for _ in range(size)]
            for _ in range(size)
        ]
        rewards[key] = [
            [0.0 for _ in range(size)]
            for _ in range(size)
        ]

    core_transitions[0] = [
        [[[1 for _ in range(5)] for _ in range(4)] for _ in range(size)]
        for _ in range(size)
    ]
    core_rewards[0] = [
        [0.0 for _ in range(size)]
        for _ in range(size)
    ]

    return policy_lib, transitions, rewards, core_transitions, core_rewards


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
    direct_dict = {
        (0, 0): 0,
        (0, 1): 1,
        (0, -1): 2,
        (-1, 0): 3,
        (1, 0): 4,
    }

    total_reward = 0

    init_x = np.random.random_integers(0, size - 1, 1)
    init_y = np.random.random_integers(0, size - 1, 1)

    x = int(init_x[0])
    y = int(init_y[0])

    optimal_mu = get_optimal_mu(size, good_acts, ind_act, ind_size, x, y, T)

    agg_memory = []
    behavior_pol = []

    regret = []
    t = 1
    i = 0

    n = {}
    mu_hat = {}
    R = {}
    K = {}

    policy_lib, transitions, rewards, core_transitions, core_rewards
        = initialize_params(policy_lib)

    horizon = T

    while t <= horizon:
        t_i = 0
        T_i = 2 ** i
        H_hat = span(T_i)
        i += 1

        gridworld = GridWorld(good_acts, size)

        while t_i <= T_i and t <= horizon and len(policy_lib) != 0:
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

            while t_i <= T_i and t <= horizon and v[m_B] <= n[m_B] \
                and mu_hat[m_B] - R[m_B] / (n[m_B] + v[m_B]) <= c[m_B] \
                    + complex_bound(H_hat, t, delta, n[m_B], 0, K[m_B]):
                t_i += 1

                policy = policy_lib[m_B]

                behavior_pol = update_behavior_pol(
                    behavior_pol,
                    policy,
                    t,
                    size,
                )

                # sample stochastic policy
                move_direction = np.random.choice(4, 1, p=policy[x][y])[0] + 1
                while policy[x][y][move_direction - 1] <= 0.00005:
                    move_direction = np.random.choice(
                        4, 1, p=policy[x][y])[0] + 1

                x_new, y_new, r = gridworld.take_action(x, y, move_direction)

                agg_memory += [(x, y, move_direction, r)]
                memory += [(x, y, move_direction, r)]

                movement = (x_new - x, y_new - y)
                if method == 'robust_dp' or method == 'dp':
                    rewards[m_B][x][y] = r
                    core_rewards[0][x][y] = r
                    transitions[m_B][x][y][
                        move_direction - 1][direct_dict[movement]] += 1
                    core_transitions[0][x][y][
                        move_direction - 1][direct_dict[movement]] += 1

                x = x_new
                y = y_new

                v[max_B_key] += 1.0
                R[max_B_key] += r
                total_reward += r

                regret += [optimal_mu - float(total_reward) / float(t)]

                t += 1

            K[m_B] += 1

            n[m_B] += v[m_B]

            mu_hat[m_B] = R[m_B] / n[m_B]

            if t_i < 100:
                continue

            # Derive new policy here
            new_policy = []
            exp_val = 0
            stdev = 0

            if method == 'offpol_a3c':
                # use off-policy actor critic algorithm
                offpol_a3c = OffpolicyActorCritic(size)
                print('deriving_new_policy')
                new_policy = offpol_a3c.derive_new_policy(
                    memory,
                    policy_lib[m_B]
                )
                exp_val, stdev = importance_sampling(
                    new_policy,
                    memory,
                    policy_lib[m_B],
                )
                # if the lower bound of new policy is higher than higher
                # bound of the cur policy, replace the old with new
                if exp_val > mu_hat[m_B]:
                    print('new_policy_found')
                    index = max(policy_lib) + 1
                    policy_lib[index] = new_policy
                    mu_hat[index] = exp_val
                    B[index] = exp_val + 1.96 * stdev
                    n[index] = n[m_B]
                    K[index] = K[m_B]
                    R[index] = 0.0

            elif method == 'offpol_a3c_agg':
                # use off-policy actor critic algorithm with aggregate memory
                offpol_a3c = OffpolicyActorCritic(size)
                print('deriving_new_policy')
                new_policy = offpol_a3c.derive_new_policy(
                    agg_memory,
                    behavior_pol,
                )
                exp_val, stdev = importance_sampling(
                    new_policy,
                    agg_memory,
                    behavior_pol,
                )
                # if the lower bound of new policy is higher than higher
                # bound of the cur policy, replace the old with new
                if exp_val > mu_hat[m_B]:
                    print('new_policy_found')
                    index = max(policy_lib) + 1
                    policy_lib[index] = new_policy
                    mu_hat[index] = exp_val
                    B[index] = exp_val + 1.96 * stdev
                    n[index] = n[m_B]
                    K[index] = K[m_B]
                    R[index] = 0.0

            if mu_hat[m_B] - R[m_B] / (n[m_B] + v[m_B]) > c[m_B] \
                    + complex_bound(H_hat, t, delta, n[m_B], v[m_B], K[m_B]):
                policy_lib.pop(m_B, None)
                c.pop(m_B, None)
                B.pop(m_B, None)
                v.pop(m_B, None)
                n.pop(m_B, None)
                mu_hat.pop(m_B, None)
                R.pop(m_B, None)
                K.pop(m_B, None)
                continue

    if method == 'robust_dp' or method == 'dp':
        robust_dp = RobustDP(size)
        policy = []
        if method == 'robust_dp':
            policy = robust_dp.value_iteration(transitions, rewards)
        else:
            policy = robust_dp.value_iteration(core_transitions, core_rewards)

        while t <= T:
            # sample stochastic policy
            x, y, r = gridworld.take_action(x, y, policy[x][y] + 1)

            total_reward += r

            regret += [optimal_mu - float(total_reward) / float(t)]

            t += 1

    return regret
