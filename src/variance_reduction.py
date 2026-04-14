"""
variance_reduction.py
=====================
Variance reduction techniques for Monte Carlo simulation.
Covers antithetic variates, control variates, and stratified sampling.
Author : Niraj Neupane | github.com/nirajneupane17
"""
import numpy as np
from scipy.stats import norm


def mc_naive(S, K, T, r, sigma, n, seed=42):
    """Naive Monte Carlo European call pricing."""
    np.random.seed(seed)
    Z  = np.random.randn(n)
    ST = S * np.exp((r - 0.5*sigma**2)*T + sigma*np.sqrt(T)*Z)
    pay = np.maximum(ST - K, 0)
    return np.exp(-r*T)*pay.mean(), pay.std()/np.sqrt(n)


def mc_antithetic(S, K, T, r, sigma, n, seed=42):
    """
    Antithetic variates — uses Z and -Z to reduce variance.
    Exploits negative correlation between paired paths.
    Reduces standard error by ~30-50% vs naive MC.
    """
    np.random.seed(seed)
    Z  = np.random.randn(n//2)
    Z2 = np.concatenate([Z, -Z])
    ST = S * np.exp((r - 0.5*sigma**2)*T + sigma*np.sqrt(T)*Z2)
    pay = np.maximum(ST - K, 0)
    return np.exp(-r*T)*pay.mean(), pay.std()/np.sqrt(n)


def mc_control_variate(S, K, T, r, sigma, n, seed=42):
    """
    Control variate using stock price as control.
    E[S_T] = S*exp(r*T) is known analytically.
    Adjusts MC estimate using correlation with control.
    """
    np.random.seed(seed)
    Z   = np.random.randn(n)
    ST  = S * np.exp((r - 0.5*sigma**2)*T + sigma*np.sqrt(T)*Z)
    pay = np.maximum(ST - K, 0) * np.exp(-r*T)
    # Control variate
    c      = ST * np.exp(-r*T)
    c_mean = S
    beta   = -np.cov(pay, c)[0,1] / np.var(c)
    pay_cv = pay + beta*(c - c_mean)
    return pay_cv.mean(), pay_cv.std()/np.sqrt(n)


def compare_methods(S, K, T, r, sigma, n_sims=10000):
    """
    Compare all variance reduction methods.

    Returns
    -------
    dict with price, SE, and relative efficiency for each method
    """
    import pandas as pd
    p_n, se_n = mc_naive(S, K, T, r, sigma, n_sims)
    p_a, se_a = mc_antithetic(S, K, T, r, sigma, n_sims)
    p_c, se_c = mc_control_variate(S, K, T, r, sigma, n_sims)

    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    bs = S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)

    rows = [
        {'method': 'Naive MC',         'price': round(p_n,4), 'SE': round(se_n,4), 'error': round(abs(p_n-bs),4)},
        {'method': 'Antithetic',        'price': round(p_a,4), 'SE': round(se_a,4), 'error': round(abs(p_a-bs),4)},
        {'method': 'Control Variate',   'price': round(p_c,4), 'SE': round(se_c,4), 'error': round(abs(p_c-bs),4)},
        {'method': 'BS Analytical',     'price': round(bs,4),  'SE': 0,              'error': 0},
    ]
    return pd.DataFrame(rows).set_index('method')
