from algorithm.core.general_rlpa import general_rlpa
import matplotlib.pyplot as plt
import copy

T = 10000
size = 10
delta = 0.005

policy_1 = [[0 for _ in range(size)] for _ in range(size)]
policy_2 = [[2 for _ in range(size)] for _ in range(size)]

policy_lib = {1: policy_1, 2: policy_2}

regret_rlpa = general_rlpa(policy_lib, delta, size, T)

t = [m for m in range(T)]
plt.figure()
plt.plot(t, regret_rlpa, label='RLPA')
# plt.plot(t, regret_general_rlpa_cur, label='General_RLPA_CUR')
# plt.plot(t, regret_general_rlpa_agg, label='General_RLPA_AGG')
# plt.plot(t, regret_pr, label='Policy Reuse')
# plt.plot(t, regret_robust_dp, label='Robust DP')
# plt.plot(t, regret_dp, label='DP')
plt.xlabel('time')
plt.ylabel('regret')
plt.legend()
plt.savefig('./alg_regret_plot/result_10.png')
