from utils import (
    transform_policy_from_deterministic_to_stochastic, select_action
)
from algorithm.core.rlpa_policy_info import RLPAPolicyInfo
from algorithm.mbie_eb import MBIEEB


class GeneralRLPAAgent:

    def __init__(
        self,
        policy_lib,
        mdp,
        state,
        delta=0.005,
        beta=1.0,
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
        self.last_action = 0
        self.beta = beta

    def initialize_params(self, policy_lib):
        policy_info = {}
        for key, val in policy_lib.items():
            policy_info[key] = RLPAPolicyInfo(val)

        return policy_info

    def has_policy(self):
        return len(self.policy_info) != 0

    def compute_bounds(self, H_hat, t, delta):
        for key, _ in self.policy_info.items():
            if not self.policy_info[key].new:
                self.policy_info[key].compute_bounds(H_hat, t, delta)
            else:
                self.policy_info[key].new = False

    def compute_best_policy(self):
        max_B_key = -1
        max_B = -100.0

        for key, val in self.policy_info.items():
            if val.new:
                max_B = 0.
                max_B_key = key
            if val.B > max_B:
                max_B = val.B
                max_B_key = key

        self.current_policy = self.policy_info[max_B_key]
        self.max_B = max_B,
        self.max_B_key = max_B_key
        return max_B, max_B_key, self.current_policy

    def take_action(self):
        self.last_action \
            = select_action(self.current_policy.policy[self.state])
        return self.last_action

    def initialize_MBIEEB(self, gamma=0.9):
        self.core_transitions = {}
        for i in range(self.mdp.size):
            for j in range(self.mdp.size):
                for a in range(4):
                    self.core_transitions[((i, j), a)] = {}
                    for i_t in range(self.mdp.size):
                        for j_t in range(self.mdp.size):
                            self.core_transitions[((i, j), a)][(i_t, j_t)] = 1.

        self.rewards = {}
        for i in range(self.mdp.size):
            for j in range(self.mdp.size):
                self.rewards[(i, j)] = 0.

        self.gamma = gamma

    def update_MBIEEB(self, last_state, reward, t):
        self.core_transitions[(last_state, self.last_action)][self.state] += 1.
        if t == 1.:
            self.rewards[self.state] = reward
        else:
            self.rewards[self.state] \
                = (self.rewards[self.state] * (t - 1.) + reward) / t

    def use_MBIEEB(self, init_state, t):
        mbie_eb_agent = MBIEEB(
            self.core_transitions,
            self.rewards,
            self.mdp,
            self.beta,
            init_state,
            self.gamma,
        )

        mbie_eb_agent.compute_Q()

        _, max_up_bound_pol_index, policy = self.compute_best_policy()
        switch = False

        for state in policy.policy.keys():
            mbie_value = mbie_eb_agent.compute_low_bound_V(state)
            rlpa_policy_action = policy.policy[state].index(
                max(policy.policy[state])
            )
            rlpa_value = mbie_eb_agent.compute_policy_Q(
                state, rlpa_policy_action
            )
            if mbie_value > rlpa_value:
                switch = True
                new_policy = mbie_eb_agent.compute_policy()
                self.policy_info[max_up_bound_pol_index].policy[state] \
                    = new_policy[state]

        if switch:
            print('found new policy')

    def drop_current_policy(self):
        self.policy_info.pop(self.max_B_key)
