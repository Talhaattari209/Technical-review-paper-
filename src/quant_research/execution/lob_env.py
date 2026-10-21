import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Dict, Tuple, Optional


class LOBExecutionEnv(gym.Env):
    """Gymnasium environment simulating Limit Order Book (LOB) execution dynamics.

    Simulates order liquidation over N time steps with:
    - Queue depletion and depth replenishment
    - Temporary and permanent price impact
    - Implementation Shortfall reward function
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        total_inventory: float = 10000.0,
        n_steps: int = 10,
        mid_price: float = 100.0,
        spread: float = 0.05,
        temp_impact_param: float = 1e-4,
        perm_impact_param: float = 5e-5,
    ):
        super().__init__()
        self.total_inventory = total_inventory
        self.n_steps = n_steps
        self.initial_mid_price = mid_price
        self.spread = spread
        self.temp_impact_param = temp_impact_param
        self.perm_impact_param = perm_impact_param

        # Action: Fraction of remaining inventory to liquidate in current step [0.0, 1.0]
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)

        # Observation: [remaining_inventory_fraction, step_fraction, current_spread, normalized_price_drift]
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32)

        self.reset()

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)
        self.current_step = 0
        self.remaining_inventory = self.total_inventory
        self.current_mid_price = self.initial_mid_price
        self.cash_accumulated = 0.0
        self.trajectories = []
        return self._get_obs(), {}

    def _get_obs(self) -> np.ndarray:
        rem_frac = self.remaining_inventory / self.total_inventory
        step_frac = self.current_step / self.n_steps
        price_drift = (self.current_mid_price - self.initial_mid_price) / self.initial_mid_price
        return np.array([rem_frac, step_frac, self.spread, price_drift], dtype=np.float32)

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        action_val = float(np.clip(action[0], 0.0, 1.0))

        # On the last step, force liquidation of all remaining inventory
        if self.current_step == self.n_steps - 1:
            shares_to_sell = self.remaining_inventory
        else:
            shares_to_sell = self.remaining_inventory * action_val

        # Microstructure Impact
        # Half spread + Temporary impact + Permanent impact
        temp_slippage = self.temp_impact_param * (shares_to_sell**0.6)
        perm_impact = self.perm_impact_param * shares_to_sell

        effective_execution_price = self.current_mid_price - (self.spread / 2.0) - temp_slippage
        self.current_mid_price -= perm_impact

        trade_value = shares_to_sell * effective_execution_price
        self.cash_accumulated += trade_value
        self.remaining_inventory -= shares_to_sell
        self.current_step += 1

        terminated = bool(self.current_step >= self.n_steps or self.remaining_inventory <= 1e-4)

        # Implementation Shortfall: Benchmark is immediate arrival execution at initial mid-price
        # Benchmark value: initial_mid_price * shares_to_sell
        # Reward is negative shortfall (i.e. minimize execution slippage and impact)
        arrival_value = self.initial_mid_price * shares_to_sell
        shortfall = arrival_value - trade_value
        reward = -float(shortfall) / (self.initial_mid_price * self.total_inventory)

        info = {
            "shares_sold": shares_to_sell,
            "exec_price": effective_execution_price,
            "mid_price": self.current_mid_price,
            "shortfall": shortfall,
        }
        self.trajectories.append(info)

        return self._get_obs(), reward, terminated, False, info
