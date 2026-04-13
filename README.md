# Monte Carlo Simulation for Risk & Derivatives Pricing

Python-based Monte Carlo framework for portfolio risk 
estimation and derivatives pricing using Geometric Brownian 
Motion, correlated asset simulation, variance reduction 
techniques, and path-dependent option pricing.

---

## Overview

Monte Carlo simulation is one of the most versatile tools 
in quantitative finance — used for VaR estimation, exotic 
derivatives pricing, stress testing, and capital modeling. 
This project builds a complete Monte Carlo engine from 
scratch, covering both risk and derivatives applications.

---

## Models Implemented

**Portfolio Risk Simulation**
- Correlated multi-asset GBM simulation
- 10,000 plus simulation paths
- Monte Carlo VaR and Expected Shortfall
- Convergence analysis across sample sizes

**European Options Pricing**
- Black-Scholes Monte Carlo vs analytical
- Call and put pricing
- Greeks estimation via finite differences
- Confidence intervals for price estimates

**Path-Dependent Options**
- Asian options (arithmetic and geometric average)
- Barrier options (knock-in and knock-out)
- Lookback options (fixed and floating strike)

**Variance Reduction Techniques**
- Antithetic variates
- Control variates
- Stratified sampling
- Convergence comparison across methods

---

## Tech Stack

![Python](https://img.shields.io/badge/Python-%233670A0.svg?style=for-the-badge&logo=python&logoColor=ffdd54) ![NumPy](https://img.shields.io/badge/NumPy-%230288D1.svg?style=for-the-badge&logo=numpy&logoColor=white) ![Pandas](https://img.shields.io/badge/Pandas-%234527A0.svg?style=for-the-badge&logo=pandas&logoColor=white) ![SciPy](https://img.shields.io/badge/SciPy-%231565C0.svg?style=for-the-badge&logo=scipy&logoColor=white) ![Matplotlib](https://img.shields.io/badge/Matplotlib-%23C62828.svg?style=for-the-badge&logo=Matplotlib&logoColor=white) ![Plotly](https://img.shields.io/badge/Plotly-%2300C853.svg?style=for-the-badge&logo=plotly&logoColor=white)

---

## Project Structure

```
Monte-Carlo-Risk-Derivatives-Pricing/
│
├── data/
│   ├── returns.csv
│   └── prices.csv
│
├── notebooks/
│   ├── 01_portfolio_risk_simulation.ipynb
│   ├── 02_european_options_pricing.ipynb
│   ├── 03_path_dependent_options.ipynb
│   ├── 04_variance_reduction.ipynb
│   └── 05_convergence_analysis.ipynb
│
├── src/
│   ├── mc_simulation.py
│   ├── options_pricing.py
│   └── variance_reduction.py
│
├── results/
│   ├── 01_mc_portfolio_paths.png
│   ├── 02_mc_var_distribution.png
│   ├── 03_european_option_pricing.png
│   ├── 04_path_dependent_options.png
│   ├── 05_variance_reduction.png
│   ├── 06_convergence_analysis.png
│   ├── mc_var_es_results.csv
│   └── options_pricing_results.csv
│
└── README.md
```

---

## Key Results

- Monte Carlo ES converges within 0.5% of analytical 
  estimate at 10,000 simulation paths
- Antithetic variates reduce standard error by ~40% 
  vs naive Monte Carlo at same sample size
- Asian options price at 15 to 20% discount to 
  equivalent European options due to averaging effect
- Barrier knock-out options show significant path 
  dependency — price sensitive to volatility assumptions

---

## Applications

- Portfolio VaR and Expected Shortfall estimation
- Exotic derivatives pricing and hedging
- Regulatory stress testing and scenario analysis
- Model validation and pricing benchmarking

---

## References

- Glasserman, P. — Monte Carlo Methods in Financial Engineering
- Hull, J. — Options, Futures and Other Derivatives
- Black, F. and Scholes, M. (1973) — The Pricing of Options
