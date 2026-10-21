import numpy as np
import pandas as pd
from typing import Dict, Optional


class FrictionAccountingEngine:
    """Accurately calculates real-world transaction costs and holding frictions.

    Enforces:
    - Bid-Ask Spread: >= 1.0 bps or half-spread per roundtrip
    - Execution Slippage: Base slippage + dynamic volume impact (order size relative to ADV)
    - Exchange Commission: Flat percentage or dollar charge per transaction
    - Short Borrow Cost: Annualized borrow rate accrued per bar held short
    """

    def __init__(
        self,
        half_spread_bps: float = 1.0,
        base_slippage_bps: float = 1.0,
        commission_bps: float = 1.0,
        annual_short_borrow_rate: float = 0.02,
        bars_per_year: int = 252,
    ):
        self.half_spread = half_spread_bps / 10000.0
        self.base_slippage = base_slippage_bps / 10000.0
        self.commission = commission_bps / 10000.0
        self.borrow_cost_per_bar = annual_short_borrow_rate / bars_per_year

    def compute_turnover_friction(
        self,
        target_positions: pd.Series,
        volumes: Optional[pd.Series] = None,
        trade_size_shares: float = 1000.0,
    ) -> pd.Series:
        """Computes trading costs from turnover: spread, commission, and dynamic volume slippage."""
        turnover = target_positions.diff().abs().fillna(target_positions.abs())

        # Base cost per unit traded
        unit_friction = self.half_spread + self.commission + self.base_slippage

        if volumes is not None:
            # Dynamic impact based on order size vs 20-bar rolling ADV
            adv = volumes.rolling(window=20, min_periods=5).mean().fillna(volumes)
            participation_rate = (trade_size_shares / (adv + 1e-6)).clip(lower=0.0, upper=0.5)
            # Dynamic slippage model: slippage scales with square root of participation
            dynamic_slip = self.base_slippage * (1.0 + 10.0 * np.sqrt(participation_rate))
            total_rate = self.half_spread + self.commission + dynamic_slip
            return turnover * total_rate

        return turnover * unit_friction

    def compute_net_returns(
        self,
        raw_asset_returns: pd.Series,
        positions: pd.Series,
        volumes: Optional[pd.Series] = None,
    ) -> pd.DataFrame:
        """Calculates gross strategy returns, cost breakdowns, and net strategy returns."""
        # Positions held at t-1 earn returns over bar t
        gross_returns = positions.shift(1).fillna(0.0) * raw_asset_returns

        turnover_costs = self.compute_turnover_friction(positions, volumes)

        # Short borrow penalty on short positions
        short_borrow_penalty = (positions.shift(1) < 0).astype(float) * self.borrow_cost_per_bar

        total_costs = turnover_costs + short_borrow_penalty
        net_returns = gross_returns - total_costs

        return pd.DataFrame(
            {
                "gross_return": gross_returns,
                "turnover_costs": turnover_costs,
                "borrow_costs": short_borrow_penalty,
                "total_friction": total_costs,
                "net_return": net_returns,
            },
            index=raw_asset_returns.index,
        )
