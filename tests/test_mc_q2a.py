"""Verification tests for the Q2A Monte Carlo engine and estimators."""

from __future__ import annotations

import importlib.util
import math
import unittest
from pathlib import Path

import numpy as np
from scipy.integrate import quad
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "code" / "approach3_q2a_mc.py"
SPEC = importlib.util.spec_from_file_location("approach3_q2a_mc", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
Q2A = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(Q2A)

S0 = 100.0
K = 100.0
R = 0.03
SIGMA = 0.2
T = 1.0
M = 252
B = 1.3 * S0


def zscore(value: float, se: float, reference: float) -> float:
    return (value - reference) / se


class TestQ2AMonteCarlo(unittest.TestCase):
    def test_gbm_moments(self):
        rng = np.random.default_rng(2026)
        n = 120_000
        z = rng.standard_normal((n, M))
        stats = Q2A.stats_chunk(z, S0, K, R, SIGMA, T, M, B)
        log_return = np.log(stats["ST"] / S0)
        expected_mean = (R - 0.5 * SIGMA * SIGMA) * T
        expected_var = SIGMA * SIGMA * T
        mean = float(np.mean(log_return))
        se_mean = float(np.std(log_return, ddof=1) / math.sqrt(n))
        variance = float(np.var(log_return, ddof=1))
        se_var = float(np.sqrt(2.0 / (n - 1)) * variance)
        self.assertLess(abs(zscore(mean, se_mean, expected_mean)), 4.0)
        self.assertLess(abs(zscore(variance, se_var, expected_var)), 4.0)
        expected_st = S0 * math.exp(R * T)
        se_st = float(np.std(stats["ST"], ddof=1) / math.sqrt(n))
        self.assertLess(abs(zscore(float(np.mean(stats["ST"])), se_st, expected_st)), 4.0)

    def test_geometric_closed_form_against_numerical_integration(self):
        closed = Q2A.geometric_call_closed(S0, K, R, SIGMA, T, M)
        mean_log = closed["mean_log"]
        sqrt_var = math.sqrt(closed["var_log"])
        df = math.exp(-R * T)

        def payoff(z):
            G = math.exp(mean_log + sqrt_var * z)
            return max(G - K, 0.0) * norm.pdf(z)

        integral, _ = quad(payoff, -10.0, 10.0, limit=200)
        price = df * integral
        self.assertLess(abs(price - closed["price"]), 1e-8)
        mean_integral, _ = quad(
            lambda z: math.exp(mean_log + sqrt_var * z) * norm.pdf(z), -10.0, 10.0, limit=200
        )
        self.assertLess(abs(mean_integral - closed["mean_G"]), 1e-8)

    def test_geometric_closed_form_against_mc(self):
        rng = np.random.default_rng(7)
        n = 120_000
        z = rng.standard_normal((n, M))
        stats = Q2A.stats_chunk(z, S0, K, R, SIGMA, T, M, B)
        closed = Q2A.geometric_call_closed(S0, K, R, SIGMA, T, M)
        df = math.exp(-R * T)
        price = df * float(np.mean(stats["geometric_call"]))
        se = df * float(np.std(stats["geometric_call"], ddof=1) / math.sqrt(n))
        self.assertLess(abs(zscore(price, se, closed["price"])), 4.0)
        mean_g = float(np.mean(stats["G"]))
        se_g = float(np.std(stats["G"], ddof=1) / math.sqrt(n))
        self.assertLess(abs(zscore(mean_g, se_g, closed["mean_G"])), 4.0)

    def test_pathwise_european_delta_vega_against_closed_form(self):
        rng = np.random.default_rng(11)
        n = 120_000
        z = rng.standard_normal((n, M))
        stats = Q2A.stats_chunk(z, S0, K, R, SIGMA, T, M, B)
        closed = Q2A.bs_call(S0, K, R, SIGMA, T)
        delta = float(np.mean(stats["european_pw_delta"]))
        delta_se = float(np.std(stats["european_pw_delta"], ddof=1) / math.sqrt(n))
        vega = float(np.mean(stats["european_pw_vega"]))
        vega_se = float(np.std(stats["european_pw_vega"], ddof=1) / math.sqrt(n))
        self.assertLess(abs(zscore(delta, delta_se, closed["delta"])), 4.0)
        self.assertLess(abs(zscore(vega, vega_se, closed["vega"])), 4.0)

    def test_digital_likelihood_ratio_against_closed_form(self):
        rng = np.random.default_rng(13)
        n = 120_000
        z = rng.standard_normal((n, M))
        stats = Q2A.stats_chunk(z, S0, K, R, SIGMA, T, M, B)
        closed = Q2A.digital_call(S0, K, R, SIGMA, T)
        delta = float(np.mean(stats["digital_lr_delta"]))
        delta_se = float(np.std(stats["digital_lr_delta"], ddof=1) / math.sqrt(n))
        vega = float(np.mean(stats["digital_lr_vega"]))
        vega_se = float(np.std(stats["digital_lr_vega"], ddof=1) / math.sqrt(n))
        self.assertLess(abs(zscore(delta, delta_se, closed["delta"])), 4.0)
        self.assertLess(abs(zscore(vega, vega_se, closed["vega"])), 4.0)
        price = math.exp(-R * T) * float(np.mean(stats["digital_call"]))
        price_se = math.exp(-R * T) * float(np.std(stats["digital_call"], ddof=1) / math.sqrt(n))
        self.assertLess(abs(zscore(price, price_se, closed["price"])), 4.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
