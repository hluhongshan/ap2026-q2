"""Verification tests for the Q2B deep-hedging module."""

from __future__ import annotations

import importlib.util
import math
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "code" / "approach2_q2b_deephedge.py"
SPEC = importlib.util.spec_from_file_location("approach2_q2b_deephedge", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
Q2B = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(Q2B)

S0 = 100.0
K = 100.0
R = 0.03
SIGMA = 0.20
T = 0.25
M = 63


class TestDeepHedgeQ2B(unittest.TestCase):
    def test_empirical_cvar_matches_manual(self):
        losses = np.arange(1, 101, dtype=float)
        manual = float(np.mean(np.sort(losses)[-5:]))
        self.assertAlmostEqual(Q2B.empirical_cvar(losses, 0.95), manual, places=12)
        self.assertAlmostEqual(Q2B.empirical_cvar(losses, 0.95), 98.0, places=12)

        uneven = np.arange(37, dtype=float) ** 2
        k = math.ceil(0.10 * len(uneven))
        manual_uneven = float(np.mean(np.sort(uneven)[-k:]))
        self.assertAlmostEqual(Q2B.empirical_cvar(uneven, 0.90), manual_uneven, places=12)

        tensor = torch.tensor(losses, dtype=torch.float64)
        self.assertAlmostEqual(
            float(Q2B.empirical_cvar_torch(tensor, 0.95)), manual, places=12
        )

    def test_hand_computed_brief_and_selffin_pnl(self):
        S = np.array([100.0, 101.0, 99.0])
        deltas = np.array([0.6, 0.4])
        kappa = 0.001
        r = 0.02
        m = 2
        dt = T / m
        payoff = max(S[-1] - K, 0.0)

        disc_brief = np.array([math.exp(r * (T - 0.0)), math.exp(r * (T - dt))])
        gains_brief = float(
            np.sum(disc_brief * deltas * np.diff(S))
        )
        cost_brief = kappa * (
            abs(deltas[0] - 0.0) * S[0] + abs(deltas[1] - deltas[0]) * S[1]
        )
        expected_brief = gains_brief - cost_brief - payoff

        disc_selffin = np.array(
            [math.exp(r * (T - dt)), math.exp(r * (T - 2.0 * dt))]
        )
        gains_selffin = float(
            np.sum(disc_selffin * deltas * (S[1:] - math.exp(r * dt) * S[:-1]))
        )
        cost_selffin = kappa * (
            math.exp(r * T) * abs(deltas[0] - 0.0) * S[0]
            + math.exp(r * (T - dt)) * abs(deltas[1] - deltas[0]) * S[1]
        )
        expected_selffin = gains_selffin - cost_selffin - payoff

        self.assertAlmostEqual(
            Q2B.pnl_brief(S, deltas, kappa, r, T, K), expected_brief, places=12
        )
        self.assertAlmostEqual(
            Q2B.pnl_selffin(S, deltas, kappa, r, T, K), expected_selffin, places=12
        )

    def test_bs_daily_selffin_mean_zero_and_std_decreases(self):
        rng = np.random.default_rng(2026)
        n = 80_000
        S = Q2B.gbm_paths(rng, S0, R, SIGMA, T, M, n)
        premium, _ = Q2B.bs_call_price_delta(S0, K, R, SIGMA, T)
        premium_fv = premium * math.exp(R * T)
        stds = {}
        for rebalance_days in (21, 5, 1):
            deltas = Q2B.bs_delta_path(S, K, R, SIGMA, T, M, rebalance_days)
            pnl = Q2B.pnl_selffin(S, deltas, 0.0, R, T, K)
            hedged = pnl + premium_fv
            mean = float(np.mean(hedged))
            se = float(np.std(hedged, ddof=1) / math.sqrt(n))
            self.assertLess(abs(mean), 5.0 * se)
            stds[rebalance_days] = float(np.std(pnl, ddof=1))
        self.assertLess(stds[1], stds[5])
        self.assertLess(stds[5], stds[21])

    def test_constant_policy_reproduces_expected_pnl(self):
        delta_value = 0.37
        kappa = 0.001
        r = 0.02
        m = 2
        policy = Q2B.make_constant_policy(delta_value)
        S = np.array([[100.0, 101.0, 99.0], [100.0, 98.0, 102.0]])
        deltas = Q2B.policy_deltas(policy, S, S0, T, m)
        np.testing.assert_allclose(deltas, delta_value, atol=1e-6, rtol=0.0)

        dt = T / m
        disc_brief = np.array([math.exp(r * (T - 0.0)), math.exp(r * (T - dt))])
        disc_selffin = np.array(
            [math.exp(r * (T - dt)), math.exp(r * (T - 2.0 * dt))]
        )
        payoff = np.maximum(S[:, -1] - K, 0.0)
        expected_brief = (
            delta_value * np.sum(disc_brief * np.diff(S, axis=1), axis=1)
            - kappa * delta_value * S[:, 0]
            - payoff
        )
        expected_selffin = (
            delta_value
            * np.sum(disc_selffin * (S[:, 1:] - math.exp(r * dt) * S[:, :-1]), axis=1)
            - kappa * math.exp(r * T) * delta_value * S[:, 0]
            - payoff
        )
        np.testing.assert_allclose(
            Q2B.pnl_brief(S, deltas, kappa, r, T, K), expected_brief, atol=1e-5, rtol=0.0
        )
        np.testing.assert_allclose(
            Q2B.pnl_selffin(S, deltas, kappa, r, T, K), expected_selffin, atol=1e-5, rtol=0.0
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
