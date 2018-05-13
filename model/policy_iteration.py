import copy


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

        self.direct_dict = {
            0: [0, 0],
            1: [0, 1],
            2: [0, -1],
            3: [-1, 0],
            4: [1, 0],
        }

        self.gamma = 0.9

    '''Get transition probability
    args:
        x, y: coordinate to identify invalid actions (state)
        action: the action needs to be taken
    ret:
        probs: vector of probabilities corresponding
        5 possible moves'''
    def get_transition_prob(self, x, y, action):
        probs = [0.0 for _ in range(5)]
        if action in self.good_acts:
            for i in range(1, 5):
                if i == action:
                    if i not in self.invalid_acts[(x, y)]:
                        probs[i] = 0.85
                    else:
                        probs[0] += 0.85
                else:
                    if i not in self.invalid_acts[(x, y)]:
                        probs[i] = 0.05
                    else:
                        probs[0] += 0.05
        else:
            probs[0] = 0.85
            for i in range(1, 5):
                if i not in self.invalid_acts[(x, y)]:
                    probs[i] = 0.0375
                else:
                    probs[0] += 0.0375

        return probs

    ''' The actual policy iteration algorithm'''
    def policy_iteration(self):
        # Boolean if the policy has changed in the last iter
        changed = True
        count = 0

        while changed:
            count += 1
            # Build a temporary value matrix
            temp_value_matrix = [
                [0.0 for _ in range(self.size)]
                for _ in range(self.size)
            ]

            # Disable next iteration if nothing changed
            changed = False

            # loop through all states to make updates
            # on the temporary value matrix
            for x in range(self.size):
                for y in range(self.size):
                    # q vector of all 4 actions
                    q = [0.0 for _ in range(4)]

                    # initialize best action and best q_value
                    best_action = -1
                    best_q = -100000.0

                    # compute q value for each state and each action
                    for i in range(4):
                        # Compute transition probability according to
                        # the current behavior i
                        probs = self.get_transition_prob(x, y, i + 1)

                        # sum up values from each direction
                        for j in range(5):
                            if probs[j] == 0:
                                continue

                            # get the intended move direction
                            movement = self.direct_dict[j]

                            new_x = x + movement[0]
                            new_y = y + movement[1]

                            # get reward
                            reward = self.reward_matrix[new_x][new_y]

                            # get value of next state
                            next_state_value = self.value_matrix[new_x][new_y]

                            # add projected value to q value of current action
                            q[i] += probs[j] \
                                * self.gamma * (reward + next_state_value)

                        # Update the current best policy
                        if q[i] > best_q:
                            best_q = q[i]
                            best_action = i + 1

                    if best_action != self.policy_matrix[x][y]:
                        changed = True
                        self.policy_matrix[x][y] = best_action

                    temp_value_matrix[x][y] = best_q

            self.value_matrix = copy.deepcopy(temp_value_matrix)

        return self.policy_matrix, count
