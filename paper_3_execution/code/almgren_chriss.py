import numpy as np
from typing import List


def compute_almgren_chriss_trajectory(
    total_inventory: float,
    n_steps: int,
    risk_aversion: float = 1e-5,
    volatility: float = 0.2,
    temp_impact_param: float = 1e-4,
    perm_impact_param: float = 5e-5,
) -> np.ndarray:
    """Computes the closed-form Almgren-Chriss (2000) optimal liquidation schedule.

    Returns:
        Array of inventory targets at each time step t = 0 ... n_steps.
    """
    # Kappa parameter determining urgency of trading
    # kappa^2 ~ (lambda * sigma^2) / eta
    eta = temp_impact_param
    sigma = volatility
    lam = risk_aversion

    kappa2 = (lam * (sigma**2)) / (eta + 1e-8)
    kappa = np.sqrt(max(1e-6, kappa2))

    t_grid = np.linspace(0, 1, n_steps + 1)
    # n_j = N * sinh(kappa * (T - t)) / sinh(kappa * T)
    sinh_kappa_T = np.sinh(kappa)
    if sinh_kappa_T == 0:
        trajectory = total_inventory * (1.0 - t_grid)
    else:
        trajectory = total_inventory * np.sinh(kappa * (1.0 - t_grid)) / sinh_kappa_T

    return trajectory
