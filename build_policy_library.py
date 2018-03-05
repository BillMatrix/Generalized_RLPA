import pickle
from policy_iteration import PolicyIteration


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
size = [i for i in range(3, 11)]

for i in range(11):
    for j in range(len(size)):
        grid_world_solver = PolicyIteration(size[j], good_actions[i])

        optimal_policy = grid_world_solver.policy_iteration()
        filename = 'mdp_{0}_size_{1}'.format(str(i), str(j))
        pickle.dump(optimal_policy, open(filename, 'wb'))
