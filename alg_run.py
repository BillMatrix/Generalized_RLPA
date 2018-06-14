from model.RLPA import rlpa
from model.general_RLPA import general_rlpa
from model.policy_reuse.policy_reuse import policy_reuse
import matplotlib.pyplot as plt
import numpy as np
import pickle
import copy

T = 20000

good_actions = [
    [1],
    [2],
    [3],
    [4],
    [1, 2],
    [1, 3],
    [1, 4],
    [2, 3],
    [2, 4],
    [3, 4],
    [1, 2, 3],
]
scenarios = [
    [1, 2],
    [1, 2, 3],
    [1, 2, 3, 4],
    [2, 3, 4, 5],
    [1, 2, 3, 4, 5],
    [1, 2, 3, 4, 5, 6],
]
size = 10

for i in range(6, len(good_actions)):
    for scenario in scenarios:
        policy_lib = {}
        for k in scenario:
            policy_lib[k] = pickle.load(
                open(
                    'optimal_policies/mdp_{0}_size_{1}'.format(
                        str(k - 1), str(size)), 'rb')
            )

        temp_policy_lib = copy.deepcopy(policy_lib)
        regret_robust_dp = general_rlpa(
            temp_policy_lib,
            0.005,
            size,
            good_actions[i],
            T,
            i,
            size,
            'robust_dp',
            True,
        )

        temp_policy_lib = copy.deepcopy(policy_lib)
        regret_dp = general_rlpa(
            temp_policy_lib,
            0.005,
            size,
            good_actions[i],
            T,
            i,
            size,
            'dp',
            True,
        )

        temp_policy_lib = copy.deepcopy(policy_lib)
        regret_rlpa = rlpa(
            temp_policy_lib,
            0.005,
            size,
            good_actions[i],
            T,
            i,
            size,
        )

        temp_policy_lib = copy.deepcopy(policy_lib)
        regret_general_rlpa_cur = general_rlpa(
            temp_policy_lib,
            0.005,
            size,
            good_actions[i],
            T,
            i,
            size,
            'offpol_a3c',
            True,
        )

        # temp_policy_lib = copy.deepcopy(policy_lib)
        # regret_general_rlpa_agg = general_rlpa(
        #     temp_policy_lib,
        #     0.005,
        #     size,
        #     good_actions[i],
        #     T,
        #     i,
        #     size,
        #     'offpol_a3c_agg',
        #     True,
        # )

        regret_pr = np.array([0.0 for _ in range(T)])
        for _ in range(10):
            temp_policy_lib = copy.deepcopy(policy_lib)
            regret_policy_reuse = policy_reuse(
                temp_policy_lib,
                size,
                good_actions[i],
                i,
                size,
                T / 5000,
                5000,
            )
            regret_pr += np.array(regret_policy_reuse)
        regret_pr = (regret_pr / 10).tolist()

        t = [m for m in range(T)]
        plt.figure()
        plt.plot(t, regret_rlpa, label='RLPA')
        plt.plot(t, regret_general_rlpa_cur, label='General_RLPA_CUR')
        # plt.plot(t, regret_general_rlpa_agg, label='General_RLPA_AGG')
        plt.plot(t, regret_pr, label='Policy Reuse')
        plt.plot(t, regret_robust_dp, label='Robust DP')
        plt.plot(t, regret_dp, label='DP')
        plt.legend()
        plt.savefig(
            './alg_regret_plot/{0}_{1}_{2}.png'.format(
                str(size), str(good_actions[i]), str(scenario)
            )
        )
        # pickle.dump(
        #     regret_rlpa,
        #     open('./alg_regret/regret_rlpa_{0}_{1}_{2}.pkl'.format(
        #         str(size), str(good_actions[i]), str(scenario)
        #     ), 'wb'),
        # )
        # pickle.dump(
        #     regret_general_rlpa_cur,
        #     open('./alg_regret/regret_g_rlpa_cur_{0}_{1}_{2}.pkl'.format(
        #         str(size), str(good_actions[i]), str(scenario)
        #     ), 'wb'),
        # )
        # pickle.dump(
        #     regret_general_rlpa_agg,
        #     open('./alg_regret/regret_g_rlpa_agg_{0}_{1}_{2}.pkl'.format(
        #         str(size), str(good_actions[i]), str(scenario)
        #     ), 'wb'),
        # )
        # pickle.dump(
        #     regret_pr,
        #     open('./alg_regret/regret_pr_{0}_{1}_{2}.pkl'.format(
        #         str(size), str(good_actions[i]), str(scenario)
        #     ), 'wb'),
        # )
