class MBIEEB:

    def __init__(self, core_transitions, rewards, mdp, beta, init_state, gamma):
        self.mdp = mdp
        self.Q = {}
        for i in range(mdp.size):
            for j in range(mdp.size):
                self.Q[(i, j)] = [0. for _ in range(4)]

        self.T = {}
        for i in range(mdp.size):
            for j in range(mdp.size):
                for a in range(4):
                    self.T[((i, j), a)] = {}
                    for i_t in range(mdp.size):
                        for j_t in range(mdp.size):
                            self.T[((i, j), a)][(i_t, j_t)] \
                                = core_transitions[((i, j), a)][(i_t, j_t)] / \
                                    sum(core_transitions[((i, j), a)])

        self.R = {}
        for i in range(mdp.size):
            for j in range(mdp.size):
                self.R[(i, j)] = rewards[(i, j)]

        self.n = {}
        for i in range(mdp.size):
            for j in range(mdp.size):
                for a in range(4):
                    self.n = sum(core_transitions[((i, j), a)])

        self.beta = beta
        self.init_state = init_state
        self.gamma = gamma

    def compute_Q(self, diff=0.05):
        new_Q = self.Q.copy()
        while abs(max(new_Q[self.init_state]) - max(self.Q[self.init_state])) > diff:
            self.Q = new_Q.copy()
            for i in range(self.mdp.size):
                for j in range(self.mdp.size):
                    for a in range(4):
                        for i_t in range(self.mdp.size):
                            for j_t in range(self.mdp.size):
                                new_Q[(i, j)][a] \
                                    += self.gamma * self.T[((i, j), a)][(i_t, j_t)] \
                                        * max(self.Q[(i_t, j_t)][a])
                        new_Q[(i, j)][a] -= self.beta / sqrt(self.n[((i, j), a)])


    def compute_init_state_V(self):
        return max(self.Q[self.init_state])

    def compute_policy(self):
        policy = {}
        for i in range(mdp.size):
            for j in range(mdp.size):
                max_a = -1
                max_Q = -1000.0
                for a in range(4):
                    if self.Q[(i, j)][a] > max_Q:
                        max_a = a
                        max_Q = self.Q[(i, j)][a]
                policy[(i, j)] = max_a

        return policy
