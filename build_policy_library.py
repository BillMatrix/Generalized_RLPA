import pickle
from model.policy_iteration import PolicyIteration


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
size = 10

for i in range(len(good_actions)):
    grid_world_solver = PolicyIteration(size, good_actions[i])

    optimal_policy, conv_count = grid_world_solver.policy_iteration()
    print(
        str(good_actions[i]) + ', ' + str(size) + ', ' + str(conv_count)
    )
    filename = 'optimal_policies/mdp_{0}_size_{1}'.format(str(i), size)
    pickle.dump(optimal_policy, open(filename, 'wb'))
