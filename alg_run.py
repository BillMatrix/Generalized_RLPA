from model.RLPA import rlpa
from model.general_RLPA import general_rlpa
import matplotlib.pyplot as plt
import pickle

T = 100000

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
sizes = [i for i in range(3, 11)]
scenarios = [
    [1, 2],
    [1, 2, 3],
    [1, 2, 3, 4],
    [2, 3, 4, 5],
    [1, 2, 3, 4, 5],
    [1, 2, 3, 4, 5, 6],
]

# for i in range(len(good_actions)):
#     for j in range(len(sizes)):
#         for scenario in scenarios:
#             policy_lib = {}
#             for i in scenario:
#                 policy_lib[i] = pickle.load(
#                     open('mdp_{0}_size_{1}'.format(str(i), str(j)), 'rb')
#                 )
#             regret_rlpa = rlpa(policy_lib, 0.005, sizes[j], good_actions[i], T)

policy_lib = {}
for i in scenarios[0]:
    policy_lib[i] = pickle.load(
        open('optimal_policies/mdp_{0}_size_{1}'.format(str(i), str(0)), 'rb')
    )
regret_rlpa = rlpa(policy_lib, 0.005, sizes[0], good_actions[0], T, 0, 0)
# print(regret_rlpa)
regret_general_rlpa = general_rlpa(
    policy_lib,
    0.05,
    sizes[0],
    good_actions[0],
    T,
    0,
    0,
    'offpol_a3c',
    True,
)
# print(regret_general_rlpa)
t = [i for i in range(T)]
plt.figure()
plt.plot(t, regret_rlpa, label='RLPA')
plt.plot(t, regret_general_rlpa, label='General_RLPA')
plt.legend()
plt.show()
