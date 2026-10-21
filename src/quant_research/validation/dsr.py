import numpy as np
import scipy.stats as stats
import pandas as pd
from typing import Dict, Optional


def compute_deflated_sharpe_ratio(
    observed_sharpe: float,
    n_trials: int,
    var_sharpe_trials: float,
    sample_length_t: int,
    skewness: float = 0.0,
    kurtosis: float = 3.0,
    n_eff: Optional[float] = None,
    output_trace_csv: Optional[str] = None,
) -> Dict[str, float]:
    """Calculates Deflated Sharpe Ratio (DSR) and its p-value per Bailey & Lopez de Prado (2014).

    Parameters:
        observed_sharpe: Annualized Sharpe ratio of selected best strategy.
        n_trials: Total attempted strategy trials (primary audit number).
        var_sharpe_trials: Variance of Sharpe ratios across all tested trials.
        sample_length_t: Number of sample observations (bars/returns).
        skewness: Return distribution skewness (0.0 for normal).
        kurtosis: Return distribution kurtosis (3.0 for normal).
        n_eff: Optional effective number of independent trials (participation ratio sensitivity proxy).
        output_trace_csv: Optional path to save the step-by-step mathematical trace ledger.
    """
    if n_trials < 1:
        raise ValueError("n_trials must be >= 1")
    if sample_length_t <= 1:
        raise ValueError("sample_length_t must be > 1")

    # Euler-Mascheroni constant
    gamma = 0.57721566490153286

    # 1. Standard Error of the Sharpe ratio under non-normality (Mertens, 2002)
    sr_std_err = np.sqrt(
        (1.0 - skewness * observed_sharpe + ((kurtosis - 1.0) / 4.0) * (observed_sharpe**2))
        / (sample_length_t - 1.0)
    )

    # 2. Expected maximum Sharpe under null hypothesis of zero true skill for N_trials
    z1 = stats.norm.ppf(1.0 - 1.0 / n_trials) if n_trials > 1 else 0.0
    z2 = stats.norm.ppf(1.0 - 1.0 / (n_trials * np.e)) if n_trials > 1 else 0.0
    expected_max_sharpe = np.sqrt(max(1e-8, var_sharpe_trials)) * ((1.0 - gamma) * z1 + gamma * z2)

    # 3. DSR z-statistic
    dsr_stat = (observed_sharpe - expected_max_sharpe) / (sr_std_err + 1e-8)

    # p_value_exceedance: Prob(SR > expected_max under H0) = 1 - Phi(dsr_stat)
    p_value_exceedance = 1.0 - stats.norm.cdf(dsr_stat)
    # p_value_cdf: Prob(Z <= dsr_stat) = Phi(dsr_stat), representing DSR confidence level
    dsr_cdf = stats.norm.cdf(dsr_stat)

    traces = [
        {
            "trial_type": "PRIMARY_N_TRIALS",
            "trial_count": n_trials,
            "observed_sharpe": observed_sharpe,
            "expected_max_sharpe": expected_max_sharpe,
            "sharpe_std_err": sr_std_err,
            "dsr_z_stat": dsr_stat,
            "p_exceedance (1-Phi)": p_value_exceedance,
            "dsr_cdf (Phi)": dsr_cdf,
            "skewness": skewness,
            "kurtosis": kurtosis,
            "sample_length_t": sample_length_t,
        }
    ]

    result = {
        "dsr_stat": float(dsr_stat),
        "p_value": float(p_value_exceedance),
        "dsr_cdf": float(dsr_cdf),
        "expected_max_sharpe": float(expected_max_sharpe),
        "observed_sharpe": float(observed_sharpe),
        "n_trials_primary": int(n_trials),
        "sample_length_t": int(sample_length_t),
        "sharpe_std_err": float(sr_std_err),
    }

    # 4. Sensitivity case if n_eff provided
    if n_eff is not None and n_eff >= 1.0:
        z1_eff = stats.norm.ppf(1.0 - 1.0 / n_eff) if n_eff > 1 else 0.0
        z2_eff = stats.norm.ppf(1.0 - 1.0 / (n_eff * np.e)) if n_eff > 1 else 0.0
        exp_max_eff = np.sqrt(max(1e-8, var_sharpe_trials)) * ((1.0 - gamma) * z1_eff + gamma * z2_eff)
        dsr_stat_eff = (observed_sharpe - exp_max_eff) / (sr_std_err + 1e-8)
        p_exceed_eff = 1.0 - stats.norm.cdf(dsr_stat_eff)
        cdf_eff = stats.norm.cdf(dsr_stat_eff)

        result["n_eff_proxy"] = float(n_eff)
        result["expected_max_sharpe_neff"] = float(exp_max_eff)
        result["dsr_stat_neff"] = float(dsr_stat_eff)
        result["p_value_neff"] = float(p_exceed_eff)
        result["dsr_cdf_neff"] = float(cdf_eff)

        traces.append(
            {
                "trial_type": "SENSITIVITY_N_EFF",
                "trial_count": n_eff,
                "observed_sharpe": observed_sharpe,
                "expected_max_sharpe": exp_max_eff,
                "sharpe_std_err": sr_std_err,
                "dsr_z_stat": dsr_stat_eff,
                "p_exceedance (1-Phi)": p_exceed_eff,
                "dsr_cdf (Phi)": cdf_eff,
                "skewness": skewness,
                "kurtosis": kurtosis,
                "sample_length_t": sample_length_t,
            }
        )

    if output_trace_csv is not None:
        pd.DataFrame(traces).to_csv(output_trace_csv, index=False)

    return result


def compute_benjamini_hochberg_fdr(p_values: np.ndarray, alpha: float = 0.05) -> Dict[str, object]:
    """Applies Benjamini-Hochberg False Discovery Rate (FDR) control."""
    m = len(p_values)
    if m == 0:
        return {"n_discoveries": 0, "rejected": np.array([])}

    sorted_indices = np.argsort(p_values)
    sorted_p = p_values[sorted_indices]

    # Threshold: p(k) <= (k / m) * alpha
    k_ranks = np.arange(1, m + 1)
    thresholds = (k_ranks / m) * alpha

    passed = sorted_p <= thresholds
    if np.any(passed):
        max_k = np.max(np.where(passed)[0])
        cutoff_p = sorted_p[max_k]
        rejected = p_values <= cutoff_p
    else:
        rejected = np.zeros(m, dtype=bool)

    return {
        "n_discoveries": int(np.sum(rejected)),
        "alpha": float(alpha),
        "rejected_mask": rejected,
        "n_tests": int(m),
    }
