from utils import move, assign_probs


class GridWorld:
    '''GridWord initialization:
        args:
            good_acts: indicating what are good actions in this girdworld.
            size: since gridworld is a square, we need the side length'''
    def __init__(self, good_acts, size):
        self.good_acts = good_acts
        self.size = size

    '''funtion used for an agent to take an action
        args:
            x, y: coordinates of the agent
            action: in {1, 2, 3, 4}, the action agent wants to take.
            1 up 2 down 3 left 4 right'''
    def take_action(self, x, y, action):
        # assign probability according to the chosen action
        # and the good action set
        probs = assign_probs(action, self.good_acts)

        # randomly select the move direction according to
        # the distribution of probabilities
        move_direction = move(probs)

        new_x = x
        new_y = y

        if move_direction == 1 and y < self.size - 1:
            new_y += 1
        elif move_direction == 2 and y > 0:
            new_y -= 1
        elif move_direction == 3 and x > 0:
            new_x -= 1
        elif move_direction == 4 and x < self.size - 1:
            new_x += 1

        if new_x == 0:
            if new_y == 0:
                return [new_x, new_y, 0.9]
            elif new_y == self.size - 1:
                return [new_x, new_y, 0.7]
        elif new_x == self.size - 1:
            if new_y == 0:
                return [new_x, new_y, 1.0]
            elif new_y == self.size - 1:
                return [new_x, new_y, 0.8]

        return [new_x, new_y, -1.0]
