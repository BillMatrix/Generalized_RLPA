import math
from utils import converge


class MBIEEB:

    def __init__(self, core_transitions, rewards, mdp, beta, init_state, gamma):
        self.mdp = mdp
        self.Q = {}
        for i in range(mdp.size):
            for j in range(mdp.size):
                self.Q[(i, j)] = [-100. for _ in range(4)]

        self.T = {}
        for i in range(mdp.size):
            for j in range(mdp.size):
                for a in range(4):
                    self.T[((i, j), a)] = {}
                    for i_t in range(mdp.size):
                        for j_t in range(mdp.size):
                            visit = sum(core_transitions[((i, j), a)].values())
                            self.T[((i, j), a)][(i_t, j_t)] = \
                                core_transitions[((i, j), a)][(i_t, j_t)] \
                                / visit

        self.R = rewards.copy()

        self.n = {}
        for i in range(mdp.size):
            for j in range(mdp.size):
                for a in range(4):
                    self.n[((i, j), a)] \
                        = sum(core_transitions[((i, j), a)].values())

        self.beta = beta
        self.init_state = init_state
        self.gamma = gamma

    def compute_Q(self, diff=0.000005, max_iter=100000):
        new_Q = {}
        for i in range(self.mdp.size):
            for j in range(self.mdp.size):
                new_Q[(i, j)] = [0. for k in range(4)]
        num_iter = 0
        while not converge(new_Q, self.Q, diff):
            num_iter += 1
            if num_iter >= max_iter:
                print('max iteration reached')
                break
            if num_iter % 1000 == 0:
                print('computing Q')
            self.Q = new_Q.copy()
            new_Q = {}
            for i in range(self.mdp.size):
                for j in range(self.mdp.size):
                    new_Q[(i, j)] = [0. for k in range(4)]

            for i in range(self.mdp.size):
                for j in range(self.mdp.size):
                    for a in range(4):
                        for i_t in range(self.mdp.size):
                            for j_t in range(self.mdp.size):
                                new_Q[(i, j)][a] \
                                    += self.gamma \
                                    * self.T[((i, j), a)][(i_t, j_t)] \
                                    * (max(self.Q[(i_t, j_t)])
                                        + self.R[(i_t, j_t)])
                        new_Q[(i, j)][a] \
                            -= self.beta / math.sqrt(self.n[((i, j), a)])

        print('num iteration: ', num_iter)
        self.Q = new_Q.copy()

    def compute_low_bound_V(self, state):
        return max(self.Q[state])

    def compute_policy_Q(self, state, a):
        return self.Q[state][a] + 2 * self.beta / math.sqrt(self.n[(state, a)])

    def compute_policy(self):
        policy = {}
        for i in range(self.mdp.size):
            for j in range(self.mdp.size):
                max_a = -1
                max_Q = -1000.0
                for a in range(4):
                    if self.Q[(i, j)][a] >= max_Q:
                        max_a = a
                        max_Q = self.Q[(i, j)][a]
                policy[(i, j)] = [1. * int(k == max_a) for k in range(4)]

        return policy
