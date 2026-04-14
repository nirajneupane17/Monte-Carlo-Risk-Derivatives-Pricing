"""
options_pricing.py
==================
Options pricing — Black-Scholes analytical and Monte Carlo.
Covers European, Asian, Barrier, and Lookback options.
Author : Niraj Neupane | github.com/nirajneupane17
"""
import numpy as np
from scipy.stats import norm


def bs_call(S, K, T, r, sigma):
    """Black-Scholes European call option price."""
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)


def bs_put(S, K, T, r, sigma):
    """Black-Scholes European put option price."""
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return K*np.exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1)


def bs_greeks(S, K, T, r, sigma):
    """
    Black-Scholes Greeks for European call.
    Returns dict with Delta, Gamma, Vega, Theta, Rho.
    """
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    delta = norm.cdf(d1)
    gamma = norm.pdf(d1) / (S*sigma*np.sqrt(T))
    vega  = S*norm.pdf(d1)*np.sqrt(T) / 100
    theta = (-(S*norm.pdf(d1)*sigma)/(2*np.sqrt(T)) - r*K*np.exp(-r*T)*norm.cdf(d2)) / 365
    rho   = K*T*np.exp(-r*T)*norm.cdf(d2) / 100
    return {'delta': round(delta,4), 'gamma': round(gamma,4),
            'vega':  round(vega,4),  'theta': round(theta,4), 'rho': round(rho,4)}


def mc_european(S, K, T, r, sigma, n_sims=50000, option_type='call', seed=42):
    """
    Monte Carlo European option pricing.

    Returns
    -------
    tuple (price, standard_error)
    """
    np.random.seed(seed)
    Z  = np.random.randn(n_sims)
    ST = S * np.exp((r - 0.5*sigma**2)*T + sigma*np.sqrt(T)*Z)
    if option_type == 'call':
        payoffs = np.maximum(ST - K, 0)
    else:
        payoffs = np.maximum(K - ST, 0)
    price = np.exp(-r*T) * payoffs.mean()
    se    = payoffs.std() / np.sqrt(n_sims)
    return price, se


def mc_asian(S, K, T, r, sigma, n_sims=50000, steps=252, seed=42):
    """
    Monte Carlo Asian call option (arithmetic average).
    Asian options are cheaper than European due to averaging effect.
    """
    np.random.seed(seed)
    dt  = T/steps
    Z   = np.random.randn(n_sims, steps)
    log_ret = (r - 0.5*sigma**2)*dt + sigma*np.sqrt(dt)*Z
    paths   = S * np.exp(np.cumsum(log_ret, axis=1))
    avg     = paths.mean(axis=1)
    payoffs = np.maximum(avg - K, 0)
    return np.exp(-r*T) * payoffs.mean()


def mc_barrier_ko(S, K, B, T, r, sigma, n_sims=50000, steps=252, seed=42):
    """
    Monte Carlo Up-and-Out Barrier call option.
    Option is knocked out (worthless) if price hits barrier B.
    """
    np.random.seed(seed)
    dt  = T/steps
    Z   = np.random.randn(n_sims, steps)
    log_ret = (r - 0.5*sigma**2)*dt + sigma*np.sqrt(dt)*Z
    paths   = S * np.exp(np.cumsum(log_ret, axis=1))
    knocked = (paths >= B).any(axis=1)
    ST      = paths[:, -1]
    payoffs = np.where(knocked, 0, np.maximum(ST - K, 0))
    return np.exp(-r*T) * payoffs.mean()


def mc_lookback(S, T, r, sigma, n_sims=50000, steps=252, seed=42):
    """
    Monte Carlo Lookback call option (floating strike).
    Payoff = S_T - min(S_t) — captures maximum possible gain.
    """
    np.random.seed(seed)
    dt  = T/steps
    Z   = np.random.randn(n_sims, steps)
    log_ret = (r - 0.5*sigma**2)*dt + sigma*np.sqrt(dt)*Z
    paths   = S * np.exp(np.cumsum(log_ret, axis=1))
    ST      = paths[:, -1]
    S_min   = paths.min(axis=1)
    return np.exp(-r*T) * (ST - S_min).mean()
