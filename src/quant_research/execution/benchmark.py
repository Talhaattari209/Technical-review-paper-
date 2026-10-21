import numpy as np
import pandas as pd
from typing import Dict, List
from quant_research.execution.lob_env import LOBExecutionEnv
from quant_research.execution.almgren_chriss import compute_almgren_chriss_trajectory


def run_execution_policy_benchmark(
    n_episodes: int = 50,
    total_inventory: float = 10000.0,
    n_steps: int = 10,
    random_seed: int = 42,
) -> Dict[str, object]:
    """Evaluates execution policies across identical simulated market episodes.

    Compares:
    - Linear TWAP
    - Volume-Weighted Average Price (VWAP) profile
    - Almgren-Chriss (optimal static liquidation)
    - Depth-Aware Dynamic Heuristic
    """
    rng = np.random.default_rng(random_seed)

    twap_shortfalls = []
    vwap_shortfalls = []
    ac_shortfalls = []
    dynamic_shortfalls = []

    # Typical U-shaped intraday volume profile for VWAP benchmark
    u_curve = np.array([1.5, 1.2, 0.9, 0.7, 0.6, 0.6, 0.7, 0.9, 1.3, 1.6][:n_steps])
    vwap_weights = u_curve / np.sum(u_curve)

    for ep in range(n_episodes):
        mid_price = 100.0 + rng.normal(0, 5.0)
        spread = max(0.01, 0.05 + rng.normal(0, 0.01))

        # 1. TWAP Evaluation (equal constant slices)
        env_twap = LOBExecutionEnv(total_inventory=total_inventory, n_steps=n_steps, mid_price=mid_price, spread=spread)
        env_twap.reset()
        twap_sf = 0.0
        for _ in range(n_steps):
            action = np.array([1.0 / max(1, (n_steps - env_twap.current_step))], dtype=np.float32)
            _, _, term, trunc, info = env_twap.step(action)
            twap_sf += info["shortfall"]
            if term or trunc:
                break
        twap_shortfalls.append(twap_sf)

        # 2. VWAP Profile Evaluation
        env_vwap = LOBExecutionEnv(total_inventory=total_inventory, n_steps=n_steps, mid_price=mid_price, spread=spread)
        env_vwap.reset()
        vwap_sf = 0.0
        for step in range(n_steps):
            w = vwap_weights[step]
            target_slice = total_inventory * w
            frac_of_rem = target_slice / max(1e-6, env_vwap.remaining_inventory)
            action = np.array([min(1.0, frac_of_rem)], dtype=np.float32)
            _, _, term, trunc, info = env_vwap.step(action)
            vwap_sf += info["shortfall"]
            if term or trunc:
                break
        vwap_shortfalls.append(vwap_sf)

        # 3. Almgren-Chriss Optimal Liquidation (risk-averse front-loaded trajectory)
        env_ac = LOBExecutionEnv(total_inventory=total_inventory, n_steps=n_steps, mid_price=mid_price, spread=spread)
        env_ac.reset()
        ac_traj = compute_almgren_chriss_trajectory(total_inventory, n_steps, risk_aversion=1e-3)
        ac_sf = 0.0
        for step in range(n_steps):
            target_next = ac_traj[step + 1]
            shares_to_sell = max(0.0, env_ac.remaining_inventory - target_next)
            frac_of_rem = shares_to_sell / max(1e-6, env_ac.remaining_inventory)
            action = np.array([min(1.0, frac_of_rem)], dtype=np.float32)
            _, _, term, trunc, info = env_ac.step(action)
            ac_sf += info["shortfall"]
            if term or trunc:
                break
        ac_shortfalls.append(ac_sf)

        # 4. Dynamic Depth-Aware Heuristic
        env_dyn = LOBExecutionEnv(total_inventory=total_inventory, n_steps=n_steps, mid_price=mid_price, spread=spread)
        env_dyn.reset()
        dyn_sf = 0.0
        for step in range(n_steps):
            base_frac = 1.0 / max(1, (n_steps - env_dyn.current_step))
            spread_factor = 0.05 / max(0.01, env_dyn.spread)
            adj_frac = np.clip(base_frac * spread_factor, 0.05, 0.5)
            action = np.array([adj_frac], dtype=np.float32)
            _, _, term, trunc, info = env_dyn.step(action)
            dyn_sf += info["shortfall"]
            if term or trunc:
                break
        dynamic_shortfalls.append(dyn_sf)

    return {
        "n_episodes": n_episodes,
        "environment_type": "Simulated Limit Order Book (LOB) with Temporary/Permanent Price Impact",
        "twap_mean_shortfall": float(np.mean(twap_shortfalls)),
        "twap_std_shortfall": float(np.std(twap_shortfalls)),
        "vwap_mean_shortfall": float(np.mean(vwap_shortfalls)),
        "vwap_std_shortfall": float(np.std(vwap_shortfalls)),
        "almgren_chriss_mean_shortfall": float(np.mean(ac_shortfalls)),
        "almgren_chriss_std_shortfall": float(np.std(ac_shortfalls)),
        "dynamic_heuristic_mean_shortfall": float(np.mean(dynamic_shortfalls)),
        "dynamic_heuristic_std_shortfall": float(np.std(dynamic_shortfalls)),
        "ac_improvement_over_twap_pct": float(
            (np.mean(twap_shortfalls) - np.mean(ac_shortfalls)) / np.mean(twap_shortfalls) * 100.0
        ),
    }
