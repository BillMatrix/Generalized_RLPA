from utils import (
    transform_policy_from_deterministic_to_stochastic, select_action
)
from algorithm.core.rlpa_policy_info import RLPAPolicyInfo


class GeneralRLPAAgent:

    def __init__(
        self,
        policy_lib,
        mdp,
        state,
        delta=0.005,
    ):
        policy_lib = \
            transform_policy_from_deterministic_to_stochastic(
                mdp.size,
                policy_lib
            )
        self.mdp = mdp
        self.state = state
        self.delta = delta

        self.policy_info = \
            self.initialize_params(policy_lib)

    def initialize_params(self, policy_lib):
        policy_info = {}
        for key, val in policy_lib.items():
            policy_info[key] = RLPAPolicyInfo(val)

        return policy_info

    def has_policy(self):
        return len(self.policy_info) != 0

    def compute_bounds(self, H_hat, t, delta):
        for key, _ in self.policy_info.items():
            self.policy_info[key].compute_bounds(H_hat, t, delta)

    def compute_best_policy(self):
        max_B_key = -1
        max_B = -100.0

        for key, val in self.policy_info.items():
            if val.B > max_B:
                max_B = val.B
                max_B_key = key

        self.current_policy = self.policy_info[max_B_key]
        return self.current_policy

    def take_action(self):
        return self.mdp.take_action(
            self.state,
            select_action(self.current_policy.policy[self.state])
        )
