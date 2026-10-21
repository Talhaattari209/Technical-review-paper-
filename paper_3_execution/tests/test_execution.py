import numpy as np
import pytest
from quant_research.execution.lob_env import LOBExecutionEnv
from quant_research.execution.almgren_chriss import compute_almgren_chriss_trajectory


def test_lob_env_execution():
    env = LOBExecutionEnv(total_inventory=1000.0, n_steps=5, mid_price=100.0)
    obs, info = env.reset()

    assert len(obs) == 4
    assert obs[0] == 1.0  # Remaining frac

    done = False
    step_count = 0
    total_reward = 0.0
    while not done:
        action = np.array([0.2], dtype=np.float32)
        obs, reward, terminated, truncated, step_info = env.step(action)
        done = terminated or truncated
        total_reward += reward
        step_count += 1

    assert step_count == 5
    assert env.remaining_inventory <= 1e-4
    assert total_reward <= 0.0  # Shortfall incurs non-positive reward


def test_almgren_chriss_trajectory():
    total_inv = 10000.0
    n_steps = 10
    traj = compute_almgren_chriss_trajectory(total_inv, n_steps, risk_aversion=1e-4)

    assert len(traj) == n_steps + 1
    assert np.isclose(traj[0], total_inv)
    assert np.isclose(traj[-1], 0.0, atol=1e-4)
    # Monotonically non-increasing inventory
    assert np.all(np.diff(traj) <= 1e-6)
