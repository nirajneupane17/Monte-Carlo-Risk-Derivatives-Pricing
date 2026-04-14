"""
mc_simulation.py
================
Monte Carlo portfolio risk simulation using Geometric
Brownian Motion and correlated asset return simulation.
Author : Niraj Neupane | github.com/nirajneupane17
"""
import numpy as np
import pandas as pd


def simulate_gbm_portfolio(port_returns, n_sims=10000, horizon=252, S0=100, seed=42):
    """
    Simulate portfolio value paths using Geometric Brownian Motion.

    Parameters
    ----------
    port_returns : pd.Series — historical portfolio returns
    n_sims       : int       — number of simulation paths
    horizon      : int       — forecast horizon in trading days
    S0           : float     — initial portfolio value

    Returns
    -------
    np.ndarray of shape (n_sims, horizon) — simulated price paths
    """
    np.random.seed(seed)
    mu  = port_returns.mean()
    sig = port_returns.std()
    dt  = 1/252
    Z   = np.random.randn(n_sims, horizon)
    log_ret = (mu - 0.5*sig**2)*dt + sig*np.sqrt(dt)*Z
    return S0 * np.exp(np.cumsum(log_ret, axis=1))


def mc_var_es(returns, weights, n_sims=10000,
              confidence_levels=[0.90, 0.95, 0.99, 0.995], seed=42):
    """
    Monte Carlo VaR and Expected Shortfall via correlated simulation.

    Parameters
    ----------
    returns           : pd.DataFrame — multi-asset daily returns
    weights           : np.ndarray   — portfolio weights
    n_sims            : int          — number of simulations
    confidence_levels : list         — confidence levels

    Returns
    -------
    pd.DataFrame with MC VaR, MC ES, and ES/VaR ratio
    """
    np.random.seed(seed)
    sim = np.random.multivariate_normal(returns.mean().values,
                                         returns.cov().values, n_sims)
    port = sim @ weights
    rows = []
    for cl in confidence_levels:
        var_val = abs(np.percentile(port, (1-cl)*100))
        tail    = port[port <= -var_val]
        es_val  = abs(tail.mean()) if len(tail) > 0 else var_val
        rows.append({'confidence': f'{int(cl*100)}%',
                     'MC_VaR':    round(var_val, 6),
                     'MC_ES':     round(es_val,  6),
                     'ES_VaR_ratio': round(es_val/var_val, 3)})
    return pd.DataFrame(rows).set_index('confidence')


def var_convergence(sim_port, confidence_level=0.99,
                    sample_sizes=[100, 500, 1000, 2000, 5000, 10000]):
    """
    Analyse VaR estimate convergence across simulation sizes.

    Returns
    -------
    pd.DataFrame with VaR estimate at each sample size
    """
    rows = []
    for n in sample_sizes:
        v = abs(np.percentile(sim_port[:n], (1-confidence_level)*100))
        rows.append({'n_sims': n, 'VaR_estimate': round(v, 6)})
    return pd.DataFrame(rows).set_index('n_sims')
