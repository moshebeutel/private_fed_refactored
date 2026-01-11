import dp_accounting


class RdpAccountantStrategy:
    def __init__(self, orders: list[float], noise_multiplier_upper_bound: float,
                 noise_multiplier_lower_bound: float = 0.0, epsilon_proximity: float = 1e-6):
        self._orders = orders
        self._noise_multiplier_upper_bound = noise_multiplier_upper_bound
        self._noise_multiplier_lower_bound = noise_multiplier_lower_bound
        self._epsilon_proximity = epsilon_proximity

    def get_noise_multiplier(self, target_epsilon: float, target_delta: float, count: int) -> float:
        current_epsilon = 100.000
        noise_multiplier_upper_bound = self._noise_multiplier_upper_bound
        noise_multiplier_lower_bound = self._noise_multiplier_lower_bound
        noise_multiplier: float = 0.0
        while abs(current_epsilon - target_epsilon) > self._epsilon_proximity:
            noise_multiplier = (noise_multiplier_upper_bound + noise_multiplier_lower_bound) / 2.0
            event = dp_accounting.SelfComposedDpEvent(dp_accounting.GaussianDpEvent(noise_multiplier=noise_multiplier),
                                                      count=1)
            accountant = dp_accounting.rdp_privacy_accountant.RdpAccountant(orders=self._orders)
            accountant.compose(event=event, count=count)
            current_epsilon, optimal_order = accountant.get_epsilon_and_optimal_order(target_delta=target_delta)
            if current_epsilon > target_epsilon:
                noise_multiplier_lower_bound = noise_multiplier
            else:
                noise_multiplier_upper_bound = noise_multiplier
            print(
                f'current_epsilon {current_epsilon} lower bound {noise_multiplier_lower_bound} '
                f'upper bound {noise_multiplier_upper_bound}')
        return noise_multiplier
