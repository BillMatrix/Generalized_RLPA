import scipy.optimize
import math
import numpy as np


class IS:

    def __init__(
        self, mdp, new_pol, data_samples, beh_pol, core_transitions, delta=0.05
    ):
        self.new_pol = new_pol
        self.data_samples = data_samples
        self.beh_pol = beh_pol
        self.delta = delta

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

        self.means = []
        for sample in self.data_samples:
            self.means += [self.imp_sampling(sample)]

    def imp_sampling(self, sample):
        behavior_pol = self.beh_pol[sample[0]].policy

        traj = sample[1]
        total_ret = 0.0
        for pair in traj:
            total_ret += pair[1]

        total_prob = 1.0
        for pair in traj:
            prob_beh = 0.0
            for a in range(4):
                prob_beh += self.T[(pair[0], a)][pair[2]] \
                    * behavior_pol[pair[0]][a]

            if prob_beh == 0.0:
                return 0.0

            prob_new = 0.0
            for a in range(4):
                prob_new += self.T[(pair[0], a)][pair[2]] \
                    * self.new_pol[pair[0]][a]

            total_prob = total_prob * prob_new / prob_beh

        return total_ret * total_prob

    def c_function(self, c):
        n_pre = len(self.data_samples) / 20.0
        n_post = len(self.data_samples) * 19.0 / 20.0

        first_term = 0.0
        sum_sqr = 0.0
        sqr_sum = 0.0
        end = len(self.data_samples) // 20
        for mean in self.means[:end]:
            first_term += min(mean, c)
            sum_sqr += min(mean, c) ** 2
            sqr_sum += min(mean, c)
        sqr_sum = sqr_sum ** 2

        first_term = first_term / n_pre

        log_delta = math.log(2 / self.delta)

        second_term = 7 * c * log_delta / 3.0 / (n_post - 1)

        inner = log_delta / n_post * 2 / n_pre / (n_pre - 1) \
            * (n_pre * sum_sqr - sqr_sum)

        if inner < 0:
            return first_term - second_term

        third_term = math.sqrt(inner)

        return first_term - second_term - third_term

    def compute_c(self):
        x0 = np.asarray([0])
        c = scipy.optimize.fmin_cg(self.c_function, x0, disp=False)
        if c < 0:
            return 0.00001
        return c[0]

    def compute_lower_bound(self):
        c = self.compute_c()
        n_post = len(self.data_samples) * 19.0 / 20.0

        l_bound = (n_post * c) ** (-1.0)
        start = len(self.data_samples) // 20

        sum_sqr = 0.0
        sum = 0.0
        for mean in self.means[start:]:
            sum += min(mean, c)

        for mean in self.means[start:]:
            for mean_0 in self.means[start:]:
                sum_sqr += (min(mean, c) - min(mean_0, c)) ** 2

        log_delta = math.log(2 / self.delta) / (n_post - 1)

        l_bound = l_bound * (
            sum / c - 7 * n_post * log_delta / 3
            - math.sqrt(log_delta * sum_sqr)
        )

        return l_bound
