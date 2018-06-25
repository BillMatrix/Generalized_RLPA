from environment.grid_world import GridWorld
import numpy as np
from utils import span, complex_bound


class RLPAAgent:

    def __init__(
        policy_lib,
        mdp,
        state,
        delta=0.005,
    ):
        self.policy_lib = policy_lib
        self.mdp = mdp
        self.state = state
        self.delta = delta

''' RLPA Algorithm
    Refer to arxiv.org/pdf/1305.1027.pdf for more details'''
def rlpa(policy_lib, delta, size, T):
    total_reward = 0
    init_x = size / 2
    init_y = size / 2
    regret = []

    t = 1
    i = 0

    grid_world = GridWorld(size)
    agent = RLPAAgent(policy_lib, grid_world, (init_x, init_y))
