from model.new_rlpa import rlpa
from model.new_general_rlpa import general_rlpa
from model.policy_reuse.policy_reuse import policy_reuse
from optimal_mu import get_optimal_mu_new
import matplotlib.pyplot as plt
import copy

T = 10000
size = 5

policy_lib = {
    1: [[1 for i in range(size)] for j in range(size)],
    2: [[3 for i in range(size)] for j in range(size)],
}

optimal_mu = get_optimal_mu_new(size, size - 1, size - 1)

temp_policy_lib = copy.deepcopy(policy_lib)
regret_dp = general_rlpa(
    temp_policy_lib,
    0.005,
    size,
    T,
    'dp',
    True,
    optimal_mu,
)

temp_policy_lib = copy.deepcopy(policy_lib)
regret_rlpa = rlpa(
    temp_policy_lib,
    0.005,
    size,
    T,
    optimal_mu,
)

# temp_policy_lib = copy.deepcopy(policy_lib)
# regret_general_rlpa_cur = general_rlpa(
#     temp_policy_lib,
#     0.005,
#     size,
#     T,
#     'offpol_a3c',
#     True,
#     optimal_mu,
# )

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

# regret_pr = np.array([0.0 for _ in range(T)])
# for _ in range(10):
#     temp_policy_lib = copy.deepcopy(policy_lib)
#     regret_policy_reuse = policy_reuse(
#         temp_policy_lib,
#         size,
#         good_actions[i],
#         i,
#         size,
#         T / 5000,
#         5000,
#     )
#     regret_pr += np.array(regret_policy_reuse)
# regret_pr = (regret_pr / 10).tolist()

t = [m for m in range(T)]
plt.figure()
plt.plot(t, regret_rlpa, label='RLPA')
# plt.plot(t, regret_general_rlpa_cur, label='General_RLPA_CUR')
# plt.plot(t, regret_general_rlpa_agg, label='General_RLPA_AGG')
# plt.plot(t, regret_pr, label='Policy Reuse')
# plt.plot(t, regret_robust_dp, label='Robust DP')
plt.plot(t, regret_dp, label='DP')
plt.xlabel('time')
plt.ylabel('regret')
plt.legend()
plt.savefig('./alg_regret_plot/result_10.png')
