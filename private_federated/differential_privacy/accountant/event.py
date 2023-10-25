import dp_accounting


class Event:
    SENSITIVITY: float = 1.0

    def __init__(self, epsilon, delta, count, strategy):
        self._strategy = strategy
        self._noise_multiplier = strategy.get_noise_multiplier(target_epsilon=epsilon, target_delta=delta, count=count)
        # epsilon_proximity = 1e-8
        # current_epsilon = 100.000
        # alphas = [alpha / 10.0 for alpha in range(11, 10000)]
        # noise_multiplier_upper_bound = 200.0
        # noise_multiplier_lower_bound = 0.0
        # while abs(current_epsilon - epsilon) > epsilon_proximity:
        #     noise_multiplier = (noise_multiplier_upper_bound + noise_multiplier_lower_bound) / 2.0
        #     event = dp_accounting.SelfComposedDpEvent(dp_accounting.GaussianDpEvent(noise_multiplier=noise_multiplier),
        #                                               count=1)
        #     accountant = dp_accounting.rdp_privacy_accountant.RdpAccountant(orders=alphas)
        #     accountant.compose(event=event, count=count)
        #     current_epsilon, optimal_order = accountant.get_epsilon_and_optimal_order(target_delta=delta)
        #     if current_epsilon > epsilon:
        #         noise_multiplier_lower_bound = noise_multiplier
        #     else:
        #         noise_multiplier_upper_bound = noise_multiplier
        #     print(
        #         f'current_epsilon {current_epsilon} lower bound {noise_multiplier_lower_bound} '
        #         f'upper bound {noise_multiplier_upper_bound}')
        #
        # self._noise_multiplier = noise_multiplier
        self._std_noise = self._noise_multiplier * Event.SENSITIVITY

    @property
    def noise_multiplier(self):
        return self._noise_multiplier

    @property
    def std_noise(self):
        return self._std_noise
