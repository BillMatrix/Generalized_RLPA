from utils import complex_bound


class RLPAPolicyInfo:
    def __init__(self, policy):
        self.policy = policy
        self.n = 1.0
        self.mu_hat = 0.0
        self.R = 0.0
        self.K = 1.0

    def compute_bounds(self, H_hat, t, delta):
        self.c = complex_bound(H_hat, t, delta, self.n, 0, self.K)
        self.B = self.mu_hat + self.c

    def initialize_v(self):
        self.v = 1.0
