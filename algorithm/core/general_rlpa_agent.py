from utils import (
    transform_policy_from_deterministic_to_stochastic, select_action
)
from algorithm.core.rlpa_policy_info import RLPAPolicyInfo
from algorithm.mbie_eb import MBIEEB
from algorithm.importance_sampling import IS
from tqdm import tqdm


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

        self.core_transitions = {}
        for i in range(self.mdp.size):
            for j in range(self.mdp.size):
                for a in range(4):
                    self.core_transitions[((i, j), a)] = {}
                    for i_t in range(self.mdp.size):
                        for j_t in range(self.mdp.size):
                            self.core_transitions[((i, j), a)][(i_t, j_t)] = 1.

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

    def initialize_IS(self):
        self.data_samples = []

    def update_IS(self, sample):
        self.data_samples += [sample]
        for s in sample[1]:
            action = self.current_policy.policy[s[0]].index(
                max(self.current_policy.policy[s[0]])
            )
            self.core_transitions[(s[0], action)][s[2]] += 1.

    def compute_best_interpolated_pol(self):
        best_policy = None
        best_policy_value = -1000.0
        for pol_ind in tqdm(range(2 ** (self.mdp.size * self.mdp.size))):
            pol = {}
            format = '{0:0' + str(self.mdp.size * self.mdp.size) + 'b}'
            assignment = format.format(pol_ind)
            for i in range(self.mdp.size):
                for j in range(self.mdp.size):
                    if assignment[i * self.mdp.size + j] == '1':
                        pol[(i, j)] = [0., 0., 1., 0.]
                    else:
                        pol[(i, j)] = [1., 0., 0., 0.]
            is_agent = IS(
                self.mdp,
                pol,
                self.data_samples,
                self.policy_info,
                self.core_transitions,
            )
            l_bound = is_agent.compute_lower_bound()
            if l_bound > best_policy_value:
                best_policy_value = l_bound
                best_policy = pol

        if best_policy_value > self.max_B[0]:
            print('found better policy')
            self.policy_info[self.max_B_key].policy = best_policy
            self.policy_info[self.max_B_key].B = best_policy_value
