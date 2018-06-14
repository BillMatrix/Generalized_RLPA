import copy
import numpy as np


class RobustDP:

    ''' Robust Dynamic Programming '''

    def __init__(self, size):
        self.size = size

        self.value_matrix = [
            [0.0 for _ in range(size)]
            for _ in range(size)
        ]

        self.reward_matrix = [
            [0.0 for _ in range(size)]
            for _ in range(size)
        ]

        self.policy_matrix = [
            [-1 for _ in range(size)]
            for _ in range(size)
        ]

        # We use a dict here to save memory
        self.invalid_acts = {}
        for i in range(size):
            for j in range(size):
                self.invalid_acts[(i, j)] = []
                if i == 0:
                    self.invalid_acts[(i, j)] += [3]
                if i == size - 1:
                    self.invalid_acts[(i, j)] += [4]
                if j == 0:
                    self.invalid_acts[(i, j)] += [2]
                if j == size - 1:
                    self.invalid_acts[(i, j)] += [1]

        self.direct_dict = {
            0: [0, 0],
            1: [0, -1],
            2: [0, 1],
            3: [-1, 0],
            4: [1, 0],
        }

        self.eps = 0.005
        self.gamma = 0.999

    def get_minimum_exp(self, x, y, action, transitions, rewards):
        minimum_val = 1000000.0
        for key, val in transitions.items():
            cur_model = val[x][y][action]
            q_value = 0.0
            for i in range(5):
                move_direction = self.direct_dict[i]
                x_new = x + move_direction[0]
                y_new = y + move_direction[1]

                if x_new >= self.size or x_new < 0:
                    continue
                if y_new >= self.size or y_new < 0:
                    continue

                reward = rewards[key][x_new][y_new]
                V_next = self.value_matrix[x_new][y_new]
                q_value += cur_model[i] * self.gamma * (reward + V_next)
            if q_value < minimum_val:
                minimum_val = q_value
        return minimum_val

    def build_transition_set(self, transitions):
        for key, val in transitions.items():
            for i in range(self.size):
                for j in range(self.size):
                    for a in range(4):
                        transition = val[i][j][a]
                        total_count = sum(transition)
                        for t in range(5):
                            transitions[key][i][j][a][t] = \
                                transition[t] * 1.0 / total_count

        return transitions

    def value_iteration(self, transitions, rewards):
        transitions = self.build_transition_set(transitions)

        new_value_matrix = [
            [0.0 for _ in range(self.size)]
            for _ in range(self.size)
        ]
        new_policy_matrix = [
            [-1 for _ in range(self.size)]
            for _ in range(self.size)
        ]

        for i in range(self.size):
            for j in range(self.size):
                max_q = -100000000.0
                max_a = 0
                for a in range(4):
                    q_val = self.get_minimum_exp(i, j, a, transitions, rewards)
                    if q_val > max_q:
                        max_a = a
                        max_q = q_val
                new_value_matrix[i][j] = max_q
                new_policy_matrix[i][j] = max_a

        target = (1. - self.gamma) * self.eps / (4 * self.gamma)
        print(target)

        while np.linalg.norm(
            np.array(new_value_matrix).flatten()
                - np.array(self.value_matrix).flatten()
        ) > target:
            self.value_matrix = copy.deepcopy(new_value_matrix)
            self.policy_matrix = copy.deepcopy(new_policy_matrix)

            for i in range(self.size):
                for j in range(self.size):
                    max_q = -100000000.0
                    max_a = 0
                    for a in range(4):
                        q_val = self.get_minimum_exp(
                            i, j, a, transitions, rewards)
                        if q_val > max_q:
                            max_a = a
                            max_q = q_val
                    new_value_matrix[i][j] = max_q
                    new_policy_matrix[i][j] = max_a

        print(new_policy_matrix)
        return new_policy_matrix
