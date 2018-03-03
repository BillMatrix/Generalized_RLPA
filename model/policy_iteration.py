from utils import assign_probs


class PolicyIteration:

    ''' Policy Iteration initialization
        args:
            size: the size of the gridworld
            good_acts: good actions in the gridworld
        stores:
            good_actions, size, value_matrix, reward_matrix,
            policy_matrix, invalid actions (at corner)'''
    def __init__(self, size, good_acts):
        self.good_acts = good_acts
        self.size = size

        self.value_matrix = [
            [0.0 for _ in range(size)]
            for _ in range(size)
        ]

        self.reward_matrix = [
            [0.0 for _ in range(size)]
            for _ in range(size)
        ]
        self.reward_matrix[0][0] = 0.9
        self.reward_matrix[0][size - 1] = 0.7
        self.reward_matrix[size - 1][0] = 1
        self.reward_matrix[size - 1][size - 1] = 0.8

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

        self.action_dict = {
            0: [0, 0],
            1: [0, 1],
            2: [0, -1],
            3: [-1, 0],
            4: [1, 0],
        }

    '''Get transition probability
    args:
        x, y: coordinate to identify invalid actions (state)
        action: the action needs to be taken
    ret:
        probs: vector of probabilities corresponding
        5 possible moves'''
    def get_transition_prob(self, x, y, action):
        probs = [0.0 for _ in range(5)]
        
