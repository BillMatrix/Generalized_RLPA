from utils import move


class GridWorld:

    def __init__(self, size):
        self.size = size
        self.move_prob = [
            [0.8, 0.0, 0.2, 0.0, 0.0],
            [0.0, 0.8, 0.0, 0.2, 0.0],
            [0.2, 0.0, 0.8, 0.0, 0.0],
            [0.0, 0.2, 0.0, 0.8, 0.0],
        ]
        self.states = []
        for i in range(size):
            for j in range(size):
                self.states += [(i, j)]

    '''funtion used for an agent to take an action
        args:
            x, y: coordinates of the agent
            action: in {0, 1, 2, 3}, the action agent wants to take.
            0 up 1 down 2 left 3 right'''
    def take_action(self, state, action):

        # randomly select the move direction according to
        # the distribution of probabilities
        move_direction = move(self.move_prob)

        new_x = state[0]
        new_y = state[1]

        if move_direction == 0 and y > 0:
            new_y -= 1
        elif move_direction == 1 and y < self.size - 1:
            new_y += 1
        elif move_direction == 2 and x > 0:
            new_x -= 1
        elif move_direction == 3 and x < self.size - 1:
            new_x += 1

        if new_x == 0 and new_y == 0:
            return [(new_x, new_y), 100.0]
        elif new_x == 0 and new_y == self.size - 1:
            return [(new_x, new_y), 10.0]
        elif new_x == self.size - 1 and new_y == 0:
            return [(new_x, new_y), 10.0]
        elif new_x >= self.size / 2 and new_y >= self.size / 2:
            return [(new_x, new_y), -1.0]

        return [(new_x, new_y), 0.0]
