import copy


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
            1: [0, 1],
            2: [0, -1],
            3: [-1, 0],
            4: [1, 0],
        }

        self.eps = 0.005
        self.gamma = 0.9

    def get_transitions(self, x, y, action):


    def value_iteration(self):
        
