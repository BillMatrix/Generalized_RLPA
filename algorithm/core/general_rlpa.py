from environment.grid_world import GridWorld
from utils import span, complex_bound
from algorithm.core.general_rlpa_agent import GeneralRLPAAgent
from optimal_mu import get_optimal_mu


def general_rlpa(policy_lib, delta, size, T, alg='none', beta=1.0):
    total_reward = 0
    init_x = size // 2
    init_y = size // 2
    init_state = (init_x, init_y)
    regret = []

    t = 1
    i = 0

    grid_world = GridWorld(size)
    agent = GeneralRLPAAgent(policy_lib, grid_world, init_state, beta=beta)
    if alg == 'MBIE-EB':
        agent.initialize_MBIEEB()
    elif alg == 'IS':
        agent.initialize_IS()

    optimal_mu = get_optimal_mu(grid_world)

    while t <= T:
        t_i = 0
        T_i = 2 ** i
        H_hat = span(T_i)
        i += 1

        while t_i <= T_i and t <= T and agent.has_policy():
            print(t)
            agent.compute_bounds(H_hat, t, delta)
            agent.compute_best_policy()
            agent.current_policy.initialize_v()

            # use pol for static variables in an episode
            pol = agent.current_policy
            current_traj = [agent.max_B_key, []]

            while t_i <= T_i and t <= T and pol.v <= pol.n \
                and pol.mu_hat - pol.R / (pol.n + pol.v) <= pol.c + \
                    complex_bound(H_hat, t, delta, pol.n, pol.v, pol.K):
                t_i += 1

                last_state = agent.state
                agent.state, r = grid_world.take_action(
                    agent.state,
                    agent.take_action()
                )

                agent.current_policy.v += 1.0
                agent.current_policy.R += r
                total_reward += r

                if alg == 'MBIE-EB':
                    agent.update_MBIEEB(last_state, r, t)
                elif alg == 'IS':
                    current_traj[1] += [(last_state, r, agent.state)]

                regret += [optimal_mu - float(total_reward) / float(t)]

                t += 1

                if t % 5 == 0:
                    agent.state = (init_x, init_y)
                    if alg == 'IS':
                        agent.update_IS(current_traj)
                        current_traj = [agent.max_B_key, []]

            agent.current_policy.K += 1

            if pol.mu_hat - agent.current_policy.R / (pol.n + agent.current_policy.v) \
                > pol.c + complex_bound(
                    H_hat, t, agent.delta, pol.n, agent.current_policy.v, pol.K):
                agent.drop_current_policy()
            else:
                agent.current_policy.n += agent.current_policy.v
                agent.current_policy.mu_hat \
                    = agent.current_policy.R / agent.current_policy.n
            if alg == 'MBIE-EB':
                agent.use_MBIEEB(init_state, t)
            elif alg == 'IS' and t > 20:
                agent.compute_best_interpolated_pol()

            if not agent.has_policy():
                print('agent does not have policy')
                return

    return regret
