import math


def importance_sampling(new_pol, data_samples, behavior_pol):
    mu = 0.0
    var = 0.0
    T = len(data_samples)
    cur_prob = 1
    for i in range(T):
        cur_state = (data_samples[i][0], data_samples[i][1])
        cur_action = data_samples[i][2]
        prob_beh_policy_cur_action \
            = behavior_pol[cur_state[0]][cur_state[1]][cur_action - 1]
        prob_new_policy_cur_action \
            = new_pol[cur_state[0]][cur_state[1]][cur_action - 1]
        cur_prob = \
            cur_prob * prob_new_policy_cur_action / prob_beh_policy_cur_action
        r = data_samples[i][3]
        if i == 0:
            mu = cur_prob * r
            var = 0.0
        else:
            cur_mu = i * 1.0 / (i + 1) * mu + 1.0 / (i + 1) * r
            var = i * 1.0 / (i + 1) * var + (r - mu) * (r - cur_mu) / (i + 1)
            mu = cur_mu

    return mu, math.sqrt(var)
