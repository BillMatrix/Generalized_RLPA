from algorithm.core.general_rlpa import general_rlpa
import matplotlib.pyplot as plt
import copy

T = 2000
size = 4
delta = 0.005
beta = 300

policy_1 = [[0 for _ in range(size)] for _ in range(size)]
policy_2 = [[2 for _ in range(size)] for _ in range(size)]

policy_lib = {1: policy_1, 2: policy_2}

print('Original RLPA')
regret_rlpa = general_rlpa(policy_lib, delta, size, T)
print('RLPA + MBIE-EB')
regret_rlpa_mbie_eb = \
    general_rlpa(policy_lib, delta, size, T, alg='MBIE-EB', beta=beta)
print('RLPA + IS')
regret_rlpa_is = \
    general_rlpa(policy_lib, delta, size, T, alg='IS')

t = [m for m in range(T)]
plt.figure()
plt.plot(t, regret_rlpa, label='RLPA')
plt.plot(t, regret_rlpa_mbie_eb, label='General_RLPA_MBIE_' + str(beta))
plt.plot(t, regret_rlpa_is, label='General_RLPA_IS')
# plt.plot(t, regret_general_rlpa_agg, label='General_RLPA_AGG')
# plt.plot(t, regret_pr, label='Policy Reuse')
# plt.plot(t, regret_robust_dp, label='Robust DP')
# plt.plot(t, regret_dp, label='DP')
plt.xlabel('time')
plt.ylabel('regret')
plt.legend()
plt.title('')
plt.savefig('./alg_regret_plot/result_compare_is_{0}.png'.format(size))
