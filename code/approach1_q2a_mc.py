#!/usr/bin/env python3
from __future__ import annotations

import argparse
import io
import json
import math
import time
import timeit
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from rich.console import Console
from rich.table import Table
from scipy.special import ndtr
from scipy.stats import kurtosis, norm

ROOT = Path(__file__).resolve().parents[1]
CLEAN = ROOT / "data" / "clean"
DEFAULT_SHIBOR = CLEAN / "shibor.csv"
DEFAULT_OUT = ROOT / "results" / "approach3"
DATASETS = ("IF", "RB", "CU", "SC")
PILLAR_DAYS = {"ON": 1, "1W": 7, "2W": 14, "1M": 30, "3M": 90, "6M": 180, "9M": 270, "1Y": 365}
HORIZON = 1.0
MONITORING = 252
BARRIER_MULT = 1.3
CALIBRATION_YEARS = 5
PAYOFFS = ("asian", "barrier", "digital", "european")
GREEKS = {
    "european": ("european_pw_delta", "european_pw_vega"),
    "asian": ("asian_pw_delta", "asian_pw_vega"),
    "digital": ("digital_lr_delta", "digital_lr_vega"),
    "barrier": ("barrier_lr_delta", "barrier_lr_vega"),
}
FD_BUMPS = (1e-1, 1e-2, 1e-3)
STAT_KEYS = (
    "ST",
    "A",
    "maxS",
    "G",
    "european_call",
    "asian_call",
    "barrier_call",
    "digital_call",
    "geometric_call",
    "european_pw_delta",
    "european_pw_vega",
    "asian_pw_delta",
    "asian_pw_vega",
    "digital_lr_delta",
    "digital_lr_vega",
    "barrier_lr_delta",
    "barrier_lr_vega",
)
PRICE_KEYS = {
    "asian": "asian_call",
    "barrier": "barrier_call",
    "digital": "digital_call",
    "european": "european_call",
}

log = Console()
console = Console()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Q2 Option A: Monte Carlo under GBM")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--shibor", type=Path, default=DEFAULT_SHIBOR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--n-paths", type=int, default=200_000)
    parser.add_argument("--n-loop", type=int, default=20_000)
    parser.add_argument("--n-pilot", type=int, default=100_000)
    parser.add_argument("--n-ref", type=int, default=1_048_576)
    parser.add_argument("--study-n-ref", type=int, default=4_194_304)
    parser.add_argument("--n-fd", type=int, default=8192)
    parser.add_argument("--replications", type=int, default=30)
    parser.add_argument("--fd-replications", type=int, default=30)
    parser.add_argument("--convergence-replications", type=int, default=30)
    parser.add_argument("--study-dataset", choices=DATASETS, default="IF")
    parser.add_argument("--skip-study", action="store_true")
    return parser.parse_args(argv)


def to_jsonable(value):
    if isinstance(value, dict):
        return {str(key): to_jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [to_jsonable(val) for val in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (np.floating, float)):
        val = float(value)
        return val if math.isfinite(val) else None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    return value


def sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_futures(code: str) -> pd.DataFrame:
    path = CLEAN / f"futures_{code}.csv"
    frame = pd.read_csv(path, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    required = {"date", "open", "high", "low", "close", "volume", "oi"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")
    if frame["date"].duplicated().any():
        raise ValueError(f"{path} contains duplicate dates")
    price_columns = ["open", "high", "low", "close"]
    if (frame[price_columns] <= 0).any().any() or frame[price_columns].isna().any().any():
        raise ValueError(f"{path} contains nonpositive or missing prices")
    if (frame["date"].max() - frame["date"].min()).days < 5 * 365.25:
        raise ValueError(f"{path} has less than five years of history")
    return frame


def load_shibor(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    required = {"date", *PILLAR_DAYS.keys()}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")
    return frame


def build_rate_curves(shibor: pd.DataFrame):
    taus = np.array([PILLAR_DAYS[name] / 365.0 for name in PILLAR_DAYS], dtype=float)
    curves = {}
    for _, row in shibor.iterrows():
        date = pd.Timestamp(row["date"])
        levels = np.array([row[name] for name in PILLAR_DAYS], dtype=float)
        if not np.all(np.isfinite(levels)):
            continue
        zeros = np.log1p((levels / 100.0) * taus * (365.0 / 360.0)) / taus
        curves[date] = (taus, zeros)
    return curves


def zero_rate(curves, sorted_dates: np.ndarray, date: pd.Timestamp, T: float):
    idx = int(np.searchsorted(sorted_dates, np.datetime64(date), side="right") - 1)
    idx = max(idx, 0)
    curve_date = pd.Timestamp(sorted_dates[idx])
    taus, zeros = curves[curve_date]
    return float(np.interp(T, taus, zeros)), curve_date


def calibration_stats(prices: np.ndarray, dates: pd.Series) -> dict:
    dates = pd.to_datetime(pd.Series(dates)).reset_index(drop=True)
    last = dates.iloc[-1]
    start = last - pd.DateOffset(years=CALIBRATION_YEARS)
    mask = dates > start
    sub_dates = dates[mask].reset_index(drop=True)
    sub_prices = np.asarray(prices)[mask.to_numpy()]
    log_prices = np.log(sub_prices)
    returns = np.diff(log_prices)
    x = np.arange(len(log_prices), dtype=float) / 252.0
    slope, intercept = np.polyfit(x, log_prices, 1.0)
    sigma = float(np.std(returns, ddof=1) * math.sqrt(252.0))
    mu_mean = float(np.mean(returns) * 252.0)
    mu_regression = float(slope)
    daily_median = float(np.median(returns))
    mad_daily = float(np.median(np.abs(returns - daily_median)))
    sigma_mad = float(1.4826 * mad_daily * math.sqrt(252.0))
    abs_returns = np.abs(returns)
    max_index = int(np.argmax(abs_returns))
    threshold = 3.0 * mad_daily
    jump_indices = np.flatnonzero(abs_returns > threshold)
    top_indices = np.argsort(abs_returns)[-10:][::-1]
    top_jumps = [
        {
            "date": sub_dates.iloc[int(index) + 1].date().isoformat(),
            "log_return": float(returns[int(index)]),
            "abs_log_return": float(abs_returns[int(index)]),
        }
        for index in top_indices
    ]
    return {
        "window_start": sub_dates.iloc[0].date().isoformat(),
        "window_end": sub_dates.iloc[-1].date().isoformat(),
        "window_years": CALIBRATION_YEARS,
        "observations": int(len(returns)),
        "mu_annualized_mean": mu_mean,
        "mu_annualized_log_regression": mu_regression,
        "log_regression_intercept": float(intercept),
        "sigma_annualized": sigma,
        "sigma_annualized_mad": sigma_mad,
        "max_abs_log_return": float(abs_returns[max_index]),
        "max_abs_log_return_date": sub_dates.iloc[max_index + 1].date().isoformat(),
        "excess_kurtosis": float(kurtosis(returns, fisher=True, bias=False)),
        "return_mean_daily": float(np.mean(returns)),
        "return_std_daily": float(np.std(returns, ddof=1)),
        "mad_daily": mad_daily,
        "roll_jump_threshold_abs_log_return": float(threshold),
        "roll_jump_count": int(len(jump_indices)),
        "roll_jump_dates": [sub_dates.iloc[int(index) + 1].date().isoformat() for index in jump_indices],
        "top_abs_returns": top_jumps,
        "roll_jump_note": (
            "Main-contract series are not roll-adjusted. Large returns are retained in the "
            "calibration sample; they are reported here and are not silently removed."
        ),
    }


def bs_call(S0: float, K: float, r: float, sigma: float, T: float) -> dict:
    if sigma <= 0 or T <= 0:
        raise ValueError("sigma and T must be positive")
    sqrt_t = math.sqrt(T)
    d1 = (math.log(S0 / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * sqrt_t)
    d2 = d1 - sigma * sqrt_t
    df = math.exp(-r * T)
    return {
        "price": float(S0 * ndtr(d1) - K * df * ndtr(d2)),
        "delta": float(ndtr(d1)),
        "vega": float(S0 * norm.pdf(d1) * sqrt_t),
        "d1": float(d1),
        "d2": float(d2),
    }


def digital_call(S0: float, K: float, r: float, sigma: float, T: float) -> dict:
    if sigma <= 0 or T <= 0:
        raise ValueError("sigma and T must be positive")
    sqrt_t = math.sqrt(T)
    d1 = (math.log(S0 / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * sqrt_t)
    d2 = d1 - sigma * sqrt_t
    df = math.exp(-r * T)
    pdf = float(norm.pdf(d2))
    return {
        "price": float(df * ndtr(d2)),
        "delta": float(df * pdf / (S0 * sigma * sqrt_t)),
        "vega": float(-df * pdf * d1 / sigma),
        "d1": float(d1),
        "d2": float(d2),
    }


def geometric_moments(S0: float, r: float, sigma: float, T: float, m: int) -> dict:
    mean_log = math.log(S0) + (r - 0.5 * sigma * sigma) * T * (m + 1) / (2.0 * m)
    var_log = sigma * sigma * T * (m + 1) * (2 * m + 1) / (6.0 * m * m)
    return {
        "mean_log": float(mean_log),
        "var_log": float(var_log),
        "mean_G": float(math.exp(mean_log + 0.5 * var_log)),
    }


def geometric_call_closed(S0: float, K: float, r: float, sigma: float, T: float, m: int) -> dict:
    mom = geometric_moments(S0, r, sigma, T, m)
    mean_log = mom["mean_log"]
    var_log = mom["var_log"]
    sqrt_v = math.sqrt(var_log)
    d1 = (mean_log - math.log(K) + var_log) / sqrt_v
    d2 = d1 - sqrt_v
    df = math.exp(-r * T)
    price = df * (math.exp(mean_log + 0.5 * var_log) * ndtr(d1) - K * ndtr(d2))
    return {"price": float(price), "d1": float(d1), "d2": float(d2), **mom}


def stats_chunk(z: np.ndarray, S0: float, K: float, r: float, sigma: float, T: float, m: int, B: float) -> dict:
    dt = T / m
    t = np.arange(1, m + 1, dtype=float) * dt
    sqrt_dt = math.sqrt(dt)
    sqrt_t = math.sqrt(T)
    drift = (r - 0.5 * sigma * sigma) * dt
    vol = sigma * sqrt_dt
    log_s0 = math.log(S0)
    df = math.exp(-r * T)
    log_paths = log_s0 + np.cumsum(drift + vol * z, axis=1)
    S = np.exp(log_paths)
    ST = S[:, -1]
    A = S.mean(axis=1)
    G = np.exp(log_paths.mean(axis=1))
    max_s = S.max(axis=1)
    european_call = np.maximum(ST - K, 0.0)
    asian_call = np.maximum(A - K, 0.0)
    barrier_call = np.maximum(ST - K, 0.0) * (max_s < B)
    digital_call_value = (ST > K).astype(float)
    geometric_call_value = np.maximum(G - K, 0.0)
    european_pw_delta = df * (ST > K) * ST / S0
    european_pw_vega = (
        df * (ST > K) * ST * (np.log(ST / S0) - (r + 0.5 * sigma * sigma) * T) / sigma
    )
    asian_pw_delta = df * (A > K) * A / S0
    asian_vega_inner = S * (log_paths - log_s0 - (r + 0.5 * sigma * sigma) * t) / sigma
    asian_pw_vega = df * (A > K) * asian_vega_inner.mean(axis=1)
    z_terminal = z.sum(axis=1) / math.sqrt(m)
    digital_lr_delta = df * digital_call_value * z_terminal / (S0 * sigma * sqrt_t)
    digital_lr_vega = df * digital_call_value * ((z_terminal * z_terminal - 1.0) / sigma - sqrt_t * z_terminal)
    lr_vega_score = np.sum((z * z - 1.0) / sigma - sqrt_dt * z, axis=1)
    barrier_lr_delta = df * barrier_call * z[:, 0] / (S0 * sigma * sqrt_dt)
    barrier_lr_vega = df * barrier_call * lr_vega_score
    return {
        "ST": ST,
        "A": A,
        "maxS": max_s,
        "G": G,
        "european_call": european_call,
        "asian_call": asian_call,
        "barrier_call": barrier_call,
        "digital_call": digital_call_value,
        "geometric_call": geometric_call_value,
        "european_pw_delta": european_pw_delta,
        "european_pw_vega": european_pw_vega,
        "asian_pw_delta": asian_pw_delta,
        "asian_pw_vega": asian_pw_vega,
        "digital_lr_delta": digital_lr_delta,
        "digital_lr_vega": digital_lr_vega,
        "barrier_lr_delta": barrier_lr_delta,
        "barrier_lr_vega": barrier_lr_vega,
    }


def stats_from_z(
    z: np.ndarray,
    S0: float,
    K: float,
    r: float,
    sigma: float,
    T: float,
    m: int,
    B: float,
    chunk: int = 32768,
) -> dict:
    n = z.shape[0]
    out = {key: np.empty(n, dtype=float) for key in STAT_KEYS}
    for start in range(0, n, chunk):
        stop = min(start + chunk, n)
        values = stats_chunk(z[start:stop], S0, K, r, sigma, T, m, B)
        for key in STAT_KEYS:
            out[key][start:stop] = values[key]
    return out


def prices_chunk(z: np.ndarray, S0: float, K: float, r: float, sigma: float, T: float, m: int, B: float) -> dict:
    dt = T / m
    drift = (r - 0.5 * sigma * sigma) * dt
    vol = sigma * math.sqrt(dt)
    log_paths = math.log(S0) + np.cumsum(drift + vol * z, axis=1)
    S = np.exp(log_paths)
    ST = S[:, -1]
    A = S.mean(axis=1)
    max_s = S.max(axis=1)
    G = np.exp(log_paths.mean(axis=1))
    return {
        "asian": np.maximum(A - K, 0.0),
        "barrier": np.maximum(ST - K, 0.0) * (max_s < B),
        "digital": (ST > K).astype(float),
        "european": np.maximum(ST - K, 0.0),
        "geometric": np.maximum(G - K, 0.0),
    }


def vectorized_prices(
    z: np.ndarray,
    S0: float,
    K: float,
    r: float,
    sigma: float,
    T: float,
    m: int,
    B: float,
    chunk: int = 32768,
) -> dict:
    n = z.shape[0]
    keys = ("asian", "barrier", "digital", "european", "geometric")
    out = {key: np.empty(n, dtype=float) for key in keys}
    for start in range(0, n, chunk):
        stop = min(start + chunk, n)
        values = prices_chunk(z[start:stop], S0, K, r, sigma, T, m, B)
        for key in keys:
            out[key][start:stop] = values[key]
    return out


def loop_prices(
    z: np.ndarray,
    S0: float,
    K: float,
    r: float,
    sigma: float,
    T: float,
    m: int,
    B: float,
) -> dict:
    dt = T / m
    drift = (r - 0.5 * sigma * sigma) * dt
    vol = sigma * math.sqrt(dt)
    log_s0 = math.log(S0)
    n = z.shape[0]
    asian = np.empty(n, dtype=float)
    barrier = np.empty(n, dtype=float)
    digital = np.empty(n, dtype=float)
    european = np.empty(n, dtype=float)
    for i in range(n):
        log_s = log_s0
        sum_s = 0.0
        max_s = -math.inf
        for j in range(m):
            log_s += drift + vol * z[i, j]
            value = math.exp(log_s)
            sum_s += value
            if value > max_s:
                max_s = value
        ST = math.exp(log_s)
        average = sum_s / m
        asian[i] = max(average - K, 0.0)
        barrier[i] = max(ST - K, 0.0) if max_s < B else 0.0
        digital[i] = 1.0 if ST > K else 0.0
        european[i] = max(ST - K, 0.0)
    return {"asian": asian, "barrier": barrier, "digital": digital, "european": european}


def mean_se(values: np.ndarray) -> tuple[float, float]:
    n = len(values)
    return float(np.mean(values)), float(np.std(values, ddof=1) / math.sqrt(n))


def estimate(values: np.ndarray, df: float = 1.0) -> dict:
    mean, se = mean_se(values)
    return {"price": float(df * mean), "se": float(df * se), "paths": int(len(values))}


def pilot_betas(
    rng: np.random.Generator,
    S0: float,
    K: float,
    r: float,
    sigma: float,
    T: float,
    m: int,
    B: float,
    n_pilot: int,
) -> dict:
    n_pilot = max(n_pilot, 2)
    z = rng.standard_normal((n_pilot, m))
    stats = stats_from_z(z, S0, K, r, sigma, T, m, B)
    half = max(1, n_pilot // 2)
    negative = stats_from_z(-z[:half], S0, K, r, sigma, T, m, B)
    control = stats["geometric_call"]
    control_var = float(np.var(control, ddof=1))
    betas = {}
    for payoff in PAYOFFS:
        target = stats[PRICE_KEYS[payoff]]
        covariance = float(np.cov(target, control, ddof=1)[0, 1])
        betas[payoff] = covariance / control_var if control_var > 0 else 0.0
    control_pair = 0.5 * (control[:half] + negative["geometric_call"])
    control_pair_var = float(np.var(control_pair, ddof=1))
    betas_anti = {}
    for payoff in PAYOFFS:
        target_pair = 0.5 * (stats[PRICE_KEYS[payoff]][:half] + negative[PRICE_KEYS[payoff]])
        covariance = float(np.cov(target_pair, control_pair, ddof=1)[0, 1])
        betas_anti[payoff] = covariance / control_pair_var if control_pair_var > 0 else 0.0
    return {
        "betas": betas,
        "betas_antithetic": betas_anti,
        "pilot_paths": int(n_pilot),
        "pilot_pairs": int(half),
        "control_variance": control_var,
    }


def _finalize_accumulator(sums: float, sums_sq: float, count: int) -> dict:
    mean = sums / count
    variance = max(sums_sq / count - mean * mean, 0.0)
    return {"mean": float(mean), "se": float(math.sqrt(variance / count)), "paths": int(count)}


def high_precision_reference(
    rng: np.random.Generator,
    S0: float,
    K: float,
    r: float,
    sigma: float,
    T: float,
    m: int,
    B: float,
    n_ref: int,
    betas: dict | None = None,
    betas_anti: dict | None = None,
    geometric_expected: float | None = None,
    chunk: int = 32768,
) -> dict:
    naive_sums = {key: 0.0 for key in STAT_KEYS}
    naive_sums_sq = {key: 0.0 for key in STAT_KEYS}
    anti_sums = {payoff: 0.0 for payoff in PAYOFFS}
    anti_sums_sq = {payoff: 0.0 for payoff in PAYOFFS}
    cv_sums = {payoff: 0.0 for payoff in PAYOFFS}
    cv_sums_sq = {payoff: 0.0 for payoff in PAYOFFS}
    cv_anti_sums = {payoff: 0.0 for payoff in PAYOFFS}
    cv_anti_sums_sq = {payoff: 0.0 for payoff in PAYOFFS}
    naive_count = 0
    pair_count = 0
    done = 0
    while done < n_ref:
        size = min(chunk, n_ref - done)
        z = rng.standard_normal((size, m))
        values = stats_chunk(z, S0, K, r, sigma, T, m, B)
        for key in STAT_KEYS:
            naive_sums[key] += float(np.sum(values[key]))
            naive_sums_sq[key] += float(np.sum(values[key] * values[key]))
        naive_count += size
        if betas is not None and betas_anti is not None and geometric_expected is not None:
            half = size // 2
            if half > 0:
                negative = stats_chunk(-z[:half], S0, K, r, sigma, T, m, B)
                for payoff in PAYOFFS:
                    key = PRICE_KEYS[payoff]
                    y = values[key][:half]
                    y_neg = negative[key]
                    y_pair = 0.5 * (y + y_neg)
                    x = values["geometric_call"][:half]
                    x_neg = negative["geometric_call"]
                    x_pair = 0.5 * (x + x_neg)
                    cv = y - betas[payoff] * (x - geometric_expected)
                    cv_anti = y_pair - betas_anti[payoff] * (x_pair - geometric_expected)
                    anti_sums[payoff] += float(np.sum(y_pair))
                    anti_sums_sq[payoff] += float(np.sum(y_pair * y_pair))
                    cv_sums[payoff] += float(np.sum(cv))
                    cv_sums_sq[payoff] += float(np.sum(cv * cv))
                    cv_anti_sums[payoff] += float(np.sum(cv_anti))
                    cv_anti_sums_sq[payoff] += float(np.sum(cv_anti * cv_anti))
                pair_count += half
        done += size
    out = {
        "naive": {
            key: _finalize_accumulator(naive_sums[key], naive_sums_sq[key], naive_count)
            for key in STAT_KEYS
        },
        "antithetic": {},
        "control_variate": {},
        "cv_antithetic": {},
    }
    if pair_count:
        for payoff in PAYOFFS:
            out["antithetic"][payoff] = _finalize_accumulator(
                anti_sums[payoff], anti_sums_sq[payoff], pair_count
            )
            out["control_variate"][payoff] = _finalize_accumulator(
                cv_sums[payoff], cv_sums_sq[payoff], pair_count
            )
            out["cv_antithetic"][payoff] = _finalize_accumulator(
                cv_anti_sums[payoff], cv_anti_sums_sq[payoff], pair_count
            )
    return out


def reference_tables(
    S0: float, K: float, r: float, sigma: float, T: float, m: int, B: float, hp: dict
) -> tuple[dict, dict, dict]:
    euro = bs_call(S0, K, r, sigma, T)
    digital = digital_call(S0, K, r, sigma, T)
    df = math.exp(-r * T)
    choices = {}
    for payoff in ("asian", "barrier"):
        candidates = {}
        key = PRICE_KEYS[payoff]
        candidates["naive"] = hp["naive"][key]
        for name in ("antithetic", "control_variate", "cv_antithetic"):
            if payoff in hp[name]:
                candidates[name] = hp[name][payoff]
        best = min(candidates.items(), key=lambda item: item[1]["se"])
        choices[payoff] = {
            "estimator": best[0],
            "mean": best[1]["mean"] * df,
            "se": best[1]["se"] * df,
            "paths": best[1]["paths"],
        }
    values = {
        "price_european": euro["price"],
        "delta_european": euro["delta"],
        "vega_european": euro["vega"],
        "price_digital": digital["price"],
        "delta_digital": digital["delta"],
        "vega_digital": digital["vega"],
        "price_asian": choices["asian"]["mean"],
        "delta_asian": hp["naive"]["asian_pw_delta"]["mean"],
        "vega_asian": hp["naive"]["asian_pw_vega"]["mean"],
        "price_barrier": choices["barrier"]["mean"],
        "delta_barrier": hp["naive"]["barrier_lr_delta"]["mean"],
        "vega_barrier": hp["naive"]["barrier_lr_vega"]["mean"],
    }
    errors = {
        "price_european": 0.0,
        "delta_european": 0.0,
        "vega_european": 0.0,
        "price_digital": 0.0,
        "delta_digital": 0.0,
        "vega_digital": 0.0,
        "price_asian": choices["asian"]["se"],
        "delta_asian": hp["naive"]["asian_pw_delta"]["se"],
        "vega_asian": hp["naive"]["asian_pw_vega"]["se"],
        "price_barrier": choices["barrier"]["se"],
        "delta_barrier": hp["naive"]["barrier_lr_delta"]["se"],
        "vega_barrier": hp["naive"]["barrier_lr_vega"]["se"],
    }
    return values, errors, choices


def z_score(value: float, se: float, reference: float, reference_se: float = 0.0) -> float:
    denominator = math.sqrt(se * se + reference_se * reference_se)
    if denominator <= 0:
        return 0.0 if value == reference else math.inf
    return float((value - reference) / denominator)


def pricing_table(
    stats: dict,
    negative: dict,
    n: int,
    df: float,
    betas: dict,
    betas_anti: dict,
    geometric_expected: float,
) -> dict:
    half = n // 2
    result = {}
    for payoff in PAYOFFS:
        key = PRICE_KEYS[payoff]
        y = stats[key]
        y_neg = negative[key][:half]
        y_pair = 0.5 * (y[:half] + y_neg)
        x = stats["geometric_call"]
        x_neg = negative["geometric_call"][:half]
        x_pair = 0.5 * (x[:half] + x_neg)
        beta = betas[payoff]
        beta_anti = betas_anti[payoff]
        naive = estimate(y, df)
        antithetic = estimate(y_pair, df)
        antithetic["pairs"] = int(half)
        antithetic["total_paths"] = int(2 * half)
        cv_values = y - beta * (x - geometric_expected)
        cv = estimate(cv_values, df)
        cv_anti_values = y_pair - beta_anti * (x_pair - geometric_expected)
        cv_anti = estimate(cv_anti_values, df)
        cv_anti["pairs"] = int(half)
        cv_anti["total_paths"] = int(2 * half)
        naive_var = float(np.var(y, ddof=1) / n)
        anti_var = float(np.var(y_pair, ddof=1) / half)
        cv_var = float(np.var(cv_values, ddof=1) / n)
        cv_anti_var = float(np.var(cv_anti_values, ddof=1) / half)
        result[payoff] = {
            "vectorized_naive": naive,
            "antithetic": antithetic,
            "control_variate": cv,
            "cv_antithetic": cv_anti,
            "variance": {
                "naive": naive_var,
                "antithetic": anti_var,
                "control_variate": cv_var,
                "cv_antithetic": cv_anti_var,
            },
            "variance_ratio_vs_naive": {
                "antithetic": naive_var / anti_var if anti_var > 0 else None,
                "control_variate": naive_var / cv_var if cv_var > 0 else None,
                "cv_antithetic": naive_var / cv_anti_var if cv_anti_var > 0 else None,
            },
            "beta": float(beta),
            "beta_antithetic": float(beta_anti),
        }
    return result


def greek_table(stats: dict, n: int) -> dict:
    out = {}
    for product, (delta_key, vega_key) in GREEKS.items():
        delta = estimate(stats[delta_key])
        vega = estimate(stats[vega_key])
        out[product] = {
            "delta": {"value": delta["price"], "se": delta["se"]},
            "vega": {"value": vega["price"], "se": vega["se"]},
            "method": "pathwise" if product in {"european", "asian"} else "likelihood_ratio",
        }
    return out


def payoffs_at_spot(base: dict, S0_base: float, S0_new: float, K: float, B: float) -> dict:
    scale = S0_new / S0_base
    ST = scale * base["ST"]
    A = scale * base["A"]
    max_s = scale * base["maxS"]
    G = scale * base["G"]
    return {
        "asian": np.maximum(A - K, 0.0),
        "barrier": np.maximum(ST - K, 0.0) * (max_s < B),
        "digital": (ST > K).astype(float),
        "european": np.maximum(ST - K, 0.0),
        "geometric": np.maximum(G - K, 0.0),
    }


def fd_bias_variance(
    rng: np.random.Generator,
    S0: float,
    K: float,
    r: float,
    sigma: float,
    T: float,
    m: int,
    B: float,
    references: dict,
    reference_errors: dict,
    n_paths: int,
    replications: int,
) -> dict:
    df = math.exp(-r * T)
    payoff_estimators = {payoff: {h: [] for h in FD_BUMPS} for payoff in PAYOFFS}
    vega_estimators = {payoff: {h: [] for h in FD_BUMPS} for payoff in PAYOFFS}
    pathwise = {}
    for product, (delta_key, vega_key) in GREEKS.items():
        pathwise[product] = {"delta": [], "vega": []}
    for _ in range(replications):
        z = rng.standard_normal((n_paths, m))
        base = stats_chunk(z, S0, K, r, sigma, T, m, B)
        for product, (delta_key, vega_key) in GREEKS.items():
            pathwise[product]["delta"].append(float(np.mean(base[delta_key])))
            pathwise[product]["vega"].append(float(np.mean(base[vega_key])))
        for h_frac in FD_BUMPS:
            h_spot = h_frac * S0
            up = payoffs_at_spot(base, S0, S0 + h_spot, K, B)
            down = payoffs_at_spot(base, S0, S0 - h_spot, K, B)
            for payoff in PAYOFFS:
                payoff_estimators[payoff][h_frac].append(
                    float(df * (np.mean(up[payoff]) - np.mean(down[payoff])) / (2.0 * h_spot))
                )
        for h_frac in FD_BUMPS:
            h_sigma = h_frac * sigma
            up = stats_chunk(z, S0, K, r, sigma + h_sigma, T, m, B)
            down = stats_chunk(z, S0, K, r, sigma - h_sigma, T, m, B)
            for payoff in PAYOFFS:
                vega_estimators[payoff][h_frac].append(
                    float(df * (np.mean(up[PRICE_KEYS[payoff]]) - np.mean(down[PRICE_KEYS[payoff]])) / (2.0 * h_sigma))
                )
    out = {"delta": {}, "vega": {}}
    rows = []
    for product in PAYOFFS:
        out["delta"][product] = {}
        out["vega"][product] = {}
        delta_ref_key = f"delta_{product}"
        vega_ref_key = f"vega_{product}"
        for h_frac in FD_BUMPS:
            delta_values = np.asarray(payoff_estimators[product][h_frac])
            vega_values = np.asarray(vega_estimators[product][h_frac])
            delta_summary = summarize_replications(
                delta_values, references[delta_ref_key], reference_errors[delta_ref_key]
            )
            vega_summary = summarize_replications(
                vega_values, references[vega_ref_key], reference_errors[vega_ref_key]
            )
            out["delta"][product][str(h_frac)] = delta_summary
            out["vega"][product][str(h_frac)] = vega_summary
            rows.append({"product": product, "greek": "delta", "bump": h_frac, **delta_summary})
            rows.append({"product": product, "greek": "vega", "bump": h_frac, **vega_summary})
    out["pathwise_or_lr"] = {}
    for product in PAYOFFS:
        delta_values = np.asarray(pathwise[product]["delta"])
        vega_values = np.asarray(pathwise[product]["vega"])
        out["pathwise_or_lr"][product] = {
            "delta": summarize_replications(
                delta_values, references[f"delta_{product}"], reference_errors[f"delta_{product}"]
            ),
            "vega": summarize_replications(
                vega_values, references[f"vega_{product}"], reference_errors[f"vega_{product}"]
            ),
        }
    return {"table": out, "rows": rows, "replications": replications, "n_paths": n_paths, "df": df}


def summarize_replications(values: np.ndarray, reference: float, reference_se: float) -> dict:
    mean = float(np.mean(values))
    se = float(np.std(values, ddof=1) / math.sqrt(len(values)))
    variance = float(np.var(values, ddof=1))
    bias = mean - reference
    rmse = float(math.sqrt(np.mean((values - reference) ** 2)))
    return {
        "mean": mean,
        "se": se,
        "variance": variance,
        "bias": float(bias),
        "rmse": rmse,
        "reference": float(reference),
        "reference_se": float(reference_se),
        "z": z_score(mean, se, reference, reference_se),
        "replications": int(len(values)),
    }


def convergence_study(
    rng: np.random.Generator,
    S0: float,
    K: float,
    r: float,
    sigma: float,
    T: float,
    m: int,
    B: float,
    references: dict,
    reference_errors: dict,
    betas: dict,
    betas_anti: dict,
    geometric_expected: float,
    n_values: list[int],
    replications: int,
) -> dict:
    df = math.exp(-r * T)
    price_estimators = {
        payoff: {name: {n: [] for n in n_values} for name in ("naive", "antithetic", "control_variate", "cv_antithetic")}
        for payoff in PAYOFFS
    }
    greek_estimators = {
        product: {greek: {n: [] for n in n_values} for greek in ("delta", "vega")}
        for product in PAYOFFS
    }
    for n_paths in n_values:
        half = n_paths // 2
        for _ in range(replications):
            z = rng.standard_normal((n_paths, m))
            stats = stats_from_z(z, S0, K, r, sigma, T, m, B)
            negative = stats_from_z(-z[:half], S0, K, r, sigma, T, m, B)
            for payoff in PAYOFFS:
                key = PRICE_KEYS[payoff]
                y = stats[key]
                y_neg = negative[key]
                y_pair = 0.5 * (y[:half] + y_neg)
                x = stats["geometric_call"]
                x_neg = negative["geometric_call"]
                x_pair = 0.5 * (x[:half] + x_neg)
                price_estimators[payoff]["naive"][n_paths].append(float(df * np.mean(y)))
                price_estimators[payoff]["antithetic"][n_paths].append(float(df * np.mean(y_pair)))
                price_estimators[payoff]["control_variate"][n_paths].append(
                    float(df * np.mean(y - betas[payoff] * (x - geometric_expected)))
                )
                price_estimators[payoff]["cv_antithetic"][n_paths].append(
                    float(df * np.mean(y_pair - betas_anti[payoff] * (x_pair - geometric_expected)))
                )
            for product, (delta_key, vega_key) in GREEKS.items():
                greek_estimators[product]["delta"][n_paths].append(float(np.mean(stats[delta_key])))
                greek_estimators[product]["vega"][n_paths].append(float(np.mean(stats[vega_key])))
    out = {"pricing": {}, "greeks": {}, "fit": {}}
    for payoff in PAYOFFS:
        out["pricing"][payoff] = {}
        reference = references[f"price_{payoff}"]
        reference_se = reference_errors[f"price_{payoff}"]
        for name in ("naive", "antithetic", "control_variate", "cv_antithetic"):
            points = []
            for n_paths in n_values:
                values = np.asarray(price_estimators[payoff][name][n_paths])
                rmse = float(math.sqrt(np.mean((values - reference) ** 2)))
                points.append({"N": n_paths, "rmse": rmse, "mean": float(np.mean(values)), "se": float(np.std(values, ddof=1) / math.sqrt(len(values)))})
            fit = fit_slope(points)
            out["pricing"][payoff][name] = {
                "points": points,
                "slope": fit["slope"],
                "intercept": fit["intercept"],
                "reference": reference,
                "reference_se": reference_se,
            }
    for product in PAYOFFS:
        out["greeks"][product] = {}
        for greek in ("delta", "vega"):
            reference = references[f"{greek}_{product}"]
            reference_se = reference_errors[f"{greek}_{product}"]
            points = []
            for n_paths in n_values:
                values = np.asarray(greek_estimators[product][greek][n_paths])
                rmse = float(math.sqrt(np.mean((values - reference) ** 2)))
                points.append({"N": n_paths, "rmse": rmse, "mean": float(np.mean(values)), "se": float(np.std(values, ddof=1) / math.sqrt(len(values)))})
            fit = fit_slope(points)
            out["greeks"][product][greek] = {
                "points": points,
                "slope": fit["slope"],
                "intercept": fit["intercept"],
                "reference": reference,
                "reference_se": reference_se,
                "method": "pathwise" if product in {"european", "asian"} else "likelihood_ratio",
            }
    return out


def fit_slope(points: list[dict]) -> dict:
    x = np.log(np.array([point["N"] for point in points], dtype=float))
    y = np.log(np.array([point["rmse"] for point in points], dtype=float))
    slope, intercept = np.polyfit(x, y, 1.0)
    return {"slope": float(slope), "intercept": float(intercept)}


def performance_study(
    out_dir: Path,
    S0: float,
    K: float,
    r: float,
    sigma: float,
    T: float,
    m: int,
    B: float,
    seed: int,
) -> dict:
    z = np.random.default_rng([seed, 99]).standard_normal((2000, m))
    loop_timer = timeit.Timer(lambda: loop_prices(z, S0, K, r, sigma, T, m, B))
    vector_timer = timeit.Timer(lambda: vectorized_prices(z, S0, K, r, sigma, T, m, B))
    loop_seconds = float(loop_timer.timeit(number=3) / 3.0)
    vector_seconds = float(vector_timer.timeit(number=3) / 3.0)
    profile_z = np.random.default_rng([seed, 100]).standard_normal((500, m))
    profile_text = ""
    try:
        from line_profiler import LineProfiler

        profiler = LineProfiler()
        profiler.add_function(loop_prices)
        profiler.enable_by_count()
        loop_prices(profile_z, S0, K, r, sigma, T, m, B)
        profiler.disable_by_count()
        stream = io.StringIO()
        profiler.print_stats(stream=stream)
        profile_text = stream.getvalue()
    except Exception as exc:
        profile_text = f"line_profiler failed: {type(exc).__name__}: {exc}\n"
    profile_path = out_dir / "line_profile.txt"
    profile_path.parent.mkdir(parents=True, exist_ok=True)
    profile_path.write_text(profile_text, encoding="utf-8")
    try:
        profile_relative = str(profile_path.resolve().relative_to(ROOT))
    except ValueError:
        profile_relative = str(profile_path)
    return {
        "paths": int(z.shape[0]),
        "monitoring_dates": m,
        "loop_seconds": loop_seconds,
        "vectorized_seconds": vector_seconds,
        "speedup": loop_seconds / vector_seconds if vector_seconds > 0 else None,
        "loop_paths_per_second": z.shape[0] / loop_seconds if loop_seconds > 0 else None,
        "vectorized_paths_per_second": z.shape[0] / vector_seconds if vector_seconds > 0 else None,
        "profile_path": profile_relative,
    }


def convergence_figure(study: dict, references: dict, path: Path):
    fig = make_subplots(
        rows=2,
        cols=4,
        subplot_titles=(
            "Asian price",
            "Barrier price",
            "Digital price",
            "European price",
            "European delta/vega",
            "Asian delta/vega",
            "Digital delta/vega",
            "Barrier delta/vega",
        ),
    )
    colors = {"naive": "#1f77b4", "antithetic": "#ff7f0e", "control_variate": "#2ca02c", "cv_antithetic": "#d62728"}
    for column, payoff in enumerate(PAYOFFS, start=1):
        for name, series in study["pricing"][payoff].items():
            x = [point["N"] for point in series["points"]]
            y = [point["rmse"] for point in series["points"]]
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=y,
                    mode="lines+markers",
                    name=f"{payoff}:{name}",
                    legendgroup=name,
                    line={"color": colors[name]},
                    showlegend=column == 1,
                ),
                row=1,
                col=column,
            )
        fig.update_xaxes(type="log", row=1, col=column)
        fig.update_yaxes(type="log", row=1, col=column)
    products = ("european", "asian", "digital", "barrier")
    greek_colors = {"delta": "#1f77b4", "vega": "#ff7f0e"}
    for column, product in enumerate(products, start=1):
        for greek in ("delta", "vega"):
            series = study["greeks"][product][greek]
            x = [point["N"] for point in series["points"]]
            y = [point["rmse"] for point in series["points"]]
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=y,
                    mode="lines+markers",
                    name=f"{product}:{greek}",
                    legendgroup=greek,
                    line={"color": greek_colors[greek], "dash": "dot" if greek == "vega" else "solid"},
                    showlegend=column == 1,
                ),
                row=2,
                col=column,
            )
        fig.update_xaxes(type="log", row=2, col=column)
        fig.update_yaxes(type="log", row=2, col=column)
    fig.update_layout(title="Q2A convergence: RMSE versus number of paths", height=900, width=1400)
    fig.write_html(path, include_plotlyjs=True)


def fd_figure(fd: dict, path: Path):
    products = ("european", "asian", "digital", "barrier")
    fig = make_subplots(
        rows=2,
        cols=4,
        subplot_titles=tuple(f"{product} {greek}" for product in products for greek in ("delta", "vega")),
    )
    row = 1
    column = 1
    for product in products:
        for greek in ("delta", "vega"):
            keys = sorted(fd["table"][greek][product], key=float)
            bumps = [float(key) for key in keys]
            bias = [abs(fd["table"][greek][product][key]["bias"]) for key in keys]
            rmse = [fd["table"][greek][product][key]["rmse"] for key in keys]
            fig.add_trace(
                go.Scatter(x=bumps, y=bias, mode="lines+markers", name=f"{product}:{greek}:bias"),
                row=row,
                col=column,
            )
            fig.add_trace(
                go.Scatter(x=bumps, y=rmse, mode="lines+markers", name=f"{product}:{greek}:rmse"),
                row=row,
                col=column,
            )
            fig.update_xaxes(type="log", row=row, col=column)
            fig.update_yaxes(type="log", row=row, col=column)
            column += 1
            if column > 4:
                column = 1
                row += 1
    fig.update_layout(title="Q2A finite-difference bias and RMSE", height=900, width=1400)
    fig.write_html(path, include_plotlyjs=True)


def print_calibration(code: str, calibration: dict):
    table = Table(title=f"{code} calibration ({calibration['window_start']}..{calibration['window_end']})")
    table.add_column("Statistic")
    table.add_column("Value", justify="right")
    rows = [
        ("observations", f"{calibration['observations']:,}"),
        ("mu (mean, annualized)", f"{calibration['mu_annualized_mean']:.6f}"),
        ("mu (log regression)", f"{calibration['mu_annualized_log_regression']:.6f}"),
        ("sigma (annualized)", f"{calibration['sigma_annualized']:.6f}"),
        ("sigma (MAD robust)", f"{calibration['sigma_annualized_mad']:.6f}"),
        ("max |log return|", f"{calibration['max_abs_log_return']:.6f}"),
        ("max |return| date", calibration["max_abs_log_return_date"]),
        ("excess kurtosis", f"{calibration['excess_kurtosis']:.4f}"),
        ("roll-jump candidates", f"{calibration['roll_jump_count']}"),
    ]
    for name, value in rows:
        table.add_row(name, value)
    console.print(table)


def print_pricing(code: str, pricing: dict, loop: dict, validation: dict):
    table = Table(title=f"{code} prices and variance reduction")
    table.add_column("Payoff")
    table.add_column("Engine")
    table.add_column("Price", justify="right")
    table.add_column("SE", justify="right")
    table.add_column("Var ratio", justify="right")
    for payoff in PAYOFFS:
        item = pricing[payoff]
        for name in ("vectorized_naive", "antithetic", "control_variate", "cv_antithetic"):
            estimate_value = item[name]
            ratio = item["variance_ratio_vs_naive"].get(name, None)
            table.add_row(
                payoff,
                name,
                f"{estimate_value['price']:.6f}",
                f"{estimate_value['se']:.6f}",
                "" if ratio is None else f"{ratio:.3f}",
            )
        table.add_row(
            payoff,
            "loop_naive",
            f"{loop[payoff]['price']:.6f}",
            f"{loop[payoff]['se']:.6f}",
            "",
        )
    console.print(table)
    console.print(
        f"{code} European validation: MC={validation['mc_price']:.6f} "
        f"SE={validation['mc_se']:.6f} BS={validation['bs_price']:.6f} z={validation['z']:.3f}"
    )


def print_study(study: dict, fd: dict, performance: dict):
    table = Table(title="Convergence slopes (log RMSE vs log N)")
    table.add_column("Estimator")
    table.add_column("Slope", justify="right")
    table.add_column("Target", justify="right")
    for payoff in PAYOFFS:
        for name, item in study["pricing"][payoff].items():
            table.add_row(f"{payoff}:{name}", f"{item['slope']:.4f}", "-0.5000")
    for product in PAYOFFS:
        for greek in ("delta", "vega"):
            item = study["greeks"][product][greek]
            table.add_row(f"{product}:{greek}", f"{item['slope']:.4f}", "-0.5000")
    console.print(table)

    perf_table = Table(title="Performance")
    perf_table.add_column("Engine")
    perf_table.add_column("Seconds", justify="right")
    perf_table.add_column("Paths/s", justify="right")
    perf_table.add_row("naive Python loop", f"{performance['loop_seconds']:.6f}", f"{performance['loop_paths_per_second']:.2f}")
    perf_table.add_row("vectorized NumPy", f"{performance['vectorized_seconds']:.6f}", f"{performance['vectorized_paths_per_second']:.2f}")
    perf_table.add_row("speedup", f"{performance['speedup']:.2f}x", "-")
    console.print(perf_table)
    console.print(f"line_profiler output: {performance['profile_path']}")


def run_dataset(
    code: str,
    args: argparse.Namespace,
    shibor: pd.DataFrame,
    curves,
    sorted_dates: np.ndarray,
    seed_offset: int,
) -> tuple[dict, dict]:
    start_time = time.perf_counter()
    frame = load_futures(code)
    calibration = calibration_stats(frame["close"].to_numpy(dtype=float), frame["date"])
    sigma = calibration["sigma_annualized"]
    S0 = float(frame["close"].iloc[-1])
    valuation_date = pd.Timestamp(frame["date"].iloc[-1])
    K = S0
    B = BARRIER_MULT * S0
    r, curve_date = zero_rate(curves, sorted_dates, valuation_date, HORIZON)
    geometric = geometric_call_closed(S0, K, r, sigma, HORIZON, MONITORING)
    euro = bs_call(S0, K, r, sigma, HORIZON)
    digital = digital_call(S0, K, r, sigma, HORIZON)
    rng = np.random.default_rng([args.seed, seed_offset])

    pilot = pilot_betas(
        rng, S0, K, r, sigma, HORIZON, MONITORING, B, args.n_pilot
    )
    z = rng.standard_normal((args.n_paths, MONITORING))
    stats = stats_from_z(z, S0, K, r, sigma, HORIZON, MONITORING, B)
    negative = stats_from_z(-z[: args.n_paths // 2], S0, K, r, sigma, HORIZON, MONITORING, B)
    geometric_cf = geometric_call_closed(S0, K, r, sigma, HORIZON, MONITORING)
    geometric_expected = geometric_cf["price"] / math.exp(-r * HORIZON)
    pricing = pricing_table(
        stats,
        negative,
        args.n_paths,
        math.exp(-r * HORIZON),
        pilot["betas"],
        pilot["betas_antithetic"],
        geometric_expected,
    )
    loop_z = np.random.default_rng([args.seed, seed_offset + 100]).standard_normal((args.n_loop, MONITORING))
    loop_values = loop_prices(loop_z, S0, K, r, sigma, HORIZON, MONITORING, B)
    loop = {payoff: estimate(loop_values[payoff], math.exp(-r * HORIZON)) for payoff in PAYOFFS}
    for payoff in PAYOFFS:
        vector_price = pricing[payoff]["vectorized_naive"]
        loop_price = loop[payoff]
        pricing[payoff]["loop_naive"] = loop_price
        pricing[payoff]["engine_agreement_z"] = z_score(
            vector_price["price"],
            vector_price["se"],
            loop_price["price"],
            loop_price["se"],
        )

    bs_z = z_score(pricing["european"]["vectorized_naive"]["price"], pricing["european"]["vectorized_naive"]["se"], euro["price"])
    validation = {
        "mc_price": pricing["european"]["vectorized_naive"]["price"],
        "mc_se": pricing["european"]["vectorized_naive"]["se"],
        "bs_price": euro["price"],
        "z": bs_z,
        "passed": bool(abs(bs_z) < 3.0),
    }
    if not validation["passed"]:
        raise RuntimeError(f"{code}: European MC validation failed with z={bs_z:.4f}")

    geometric_mc_price = estimate(stats["geometric_call"], math.exp(-r * HORIZON))
    geometric_mc_mean = estimate(stats["G"])
    closed_form_checks = {
        "geometric_call_mc": geometric_mc_price,
        "geometric_call_closed": geometric_cf["price"],
        "geometric_call_z": z_score(
            geometric_mc_price["price"], geometric_mc_price["se"], geometric_cf["price"]
        ),
        "geometric_mean_mc": geometric_mc_mean,
        "geometric_mean_closed": geometric["mean_G"],
        "geometric_mean_z": z_score(
            geometric_mc_mean["price"], geometric_mc_mean["se"], geometric["mean_G"]
        ),
        "mean_log": geometric["mean_log"],
        "var_log": geometric["var_log"],
    }
    hp = high_precision_reference(
        np.random.default_rng([args.seed, seed_offset + 200]),
        S0,
        K,
        r,
        sigma,
        HORIZON,
        MONITORING,
        B,
        args.n_ref,
        betas=pilot["betas"],
        betas_anti=pilot["betas_antithetic"],
        geometric_expected=geometric_expected,
    )
    references, reference_errors, reference_choices = reference_tables(
        S0, K, r, sigma, HORIZON, MONITORING, B, hp
    )
    greeks = greek_table(stats, args.n_paths)
    for product in PAYOFFS:
        for greek in ("delta", "vega"):
            item = greeks[product][greek]
            reference = references[f"{greek}_{product}"]
            reference_se = reference_errors[f"{greek}_{product}"]
            item["reference"] = reference
            item["reference_se"] = reference_se
            item["z"] = z_score(item["value"], item["se"], reference, reference_se)
    fd = fd_bias_variance(
        np.random.default_rng([args.seed, seed_offset + 300]),
        S0,
        K,
        r,
        sigma,
        HORIZON,
        MONITORING,
        B,
        references,
        reference_errors,
        args.n_fd,
        args.fd_replications,
    )
    elapsed = time.perf_counter() - start_time
    params = {
        "approach": "Q2A",
        "dataset": code,
        "seed": args.seed,
        "input": str((CLEAN / f"futures_{code}.csv").relative_to(ROOT)),
        "input_sha256": sha256_file(CLEAN / f"futures_{code}.csv"),
        "valuation_date": valuation_date.date().isoformat(),
        "S0": S0,
        "K": K,
        "T": HORIZON,
        "m": MONITORING,
        "barrier": B,
        "barrier_multiplier": BARRIER_MULT,
        "r": r,
        "rate_convention": "SHIBOR zero from approach1: log(1 + quoted*365/360*tau)/tau, linear in tau",
        "shibor_date": curve_date.date().isoformat(),
        "shibor_sha256": sha256_file(args.shibor),
        "sigma": sigma,
        "calibration_window": {
            "start": calibration["window_start"],
            "end": calibration["window_end"],
            "years": CALIBRATION_YEARS,
        },
        "drift_note": (
            "Under Q a futures price has zero drift; the brief specifies drift r, so this "
            "implementation follows the brief's GBM drift r as a modelling simplification."
        ),
        "payoff_definitions": {
            "asian": "max((1/m) * sum_{i=1}^m S_{t_i} - K, 0)",
            "barrier": "max(S_T - K, 0) * 1{max_{i=1}^m S_{t_i} < B}",
            "digital": "1{S_T > K} (cash-or-nothing, unit payout)",
            "european": "max(S_T - K, 0)",
        },
        "pilot_paths": pilot["pilot_paths"],
        "n_paths": args.n_paths,
        "n_loop": args.n_loop,
        "n_ref": args.n_ref,
        "n_fd": args.n_fd,
        "fd_replications": args.fd_replications,
    }
    metrics = {
        "calibration": calibration,
        "rate": {"r": r, "shibor_date": curve_date.date().isoformat()},
        "validation_european": validation,
        "closed_form_checks": closed_form_checks,
        "pricing": pricing,
        "variance_reduction_benchmark_asian": {
            "antithetic": pricing["asian"]["variance_ratio_vs_naive"]["antithetic"],
            "control_variate": pricing["asian"]["variance_ratio_vs_naive"]["control_variate"],
            "cv_antithetic": pricing["asian"]["variance_ratio_vs_naive"]["cv_antithetic"],
            "passes_5x": bool(
                max(
                    pricing["asian"]["variance_ratio_vs_naive"]["antithetic"] or 0.0,
                    pricing["asian"]["variance_ratio_vs_naive"]["control_variate"] or 0.0,
                    pricing["asian"]["variance_ratio_vs_naive"]["cv_antithetic"] or 0.0,
                )
                >= 5.0
            ),
        },
        "greeks": greeks,
        "fd_bias_variance": fd,
        "references": references,
        "reference_errors": reference_errors,
        "reference_choices": reference_choices,
        "pilot": pilot,
    }
    history = {
        "timings": {"total_seconds": elapsed},
        "fd_rows": fd["rows"],
        "calibration_note": calibration["roll_jump_note"],
        "top_abs_returns": calibration["top_abs_returns"],
    }
    result = {
        "params": params,
        "metrics": metrics,
        "fit_seconds": float(elapsed),
        "history": history,
    }
    print_calibration(code, calibration)
    print_pricing(code, pricing, loop, validation)
    return result, {"pilot": pilot, "references": references, "reference_errors": reference_errors}


def run_study(
    args: argparse.Namespace,
    dataset_result: dict,
    extra: dict,
) -> dict:
    start_time = time.perf_counter()
    params = dataset_result["params"]
    S0 = params["S0"]
    K = params["K"]
    r = params["r"]
    sigma = params["sigma"]
    B = params["barrier"]
    n_values = [2**power for power in range(10, 17)]
    geometric_expected = geometric_call_closed(S0, K, r, sigma, HORIZON, MONITORING)["price"] / math.exp(-r * HORIZON)
    study_n_ref = max(args.n_ref, args.study_n_ref)
    hp_study = high_precision_reference(
        np.random.default_rng([args.seed, 901]),
        S0,
        K,
        r,
        sigma,
        HORIZON,
        MONITORING,
        B,
        study_n_ref,
        betas=extra["pilot"]["betas"],
        betas_anti=extra["pilot"]["betas_antithetic"],
        geometric_expected=geometric_expected,
    )
    study_references, study_reference_errors, study_reference_choices = reference_tables(
        S0, K, r, sigma, HORIZON, MONITORING, B, hp_study
    )
    study = convergence_study(
        np.random.default_rng([args.seed, 900]),
        S0,
        K,
        r,
        sigma,
        HORIZON,
        MONITORING,
        B,
        study_references,
        study_reference_errors,
        extra["pilot"]["betas"],
        extra["pilot"]["betas_antithetic"],
        geometric_expected,
        n_values,
        args.convergence_replications,
    )
    study_dir = args.out_dir / f"{args.study_dataset}_study"
    study_dir.mkdir(parents=True, exist_ok=True)
    performance = performance_study(
        study_dir,
        S0,
        K,
        r,
        sigma,
        HORIZON,
        MONITORING,
        B,
        args.seed,
    )
    top_profile = args.out_dir / "line_profile.txt"
    top_profile.write_text((study_dir / "line_profile.txt").read_text(encoding="utf-8"), encoding="utf-8")
    performance["profile_path_study"] = performance["profile_path"]
    try:
        performance["profile_path"] = str(top_profile.resolve().relative_to(ROOT))
    except ValueError:
        performance["profile_path"] = str(top_profile)
    elapsed = time.perf_counter() - start_time
    convergence_figure(study, study_references, study_dir / "convergence.html")
    fd_figure(dataset_result["metrics"]["fd_bias_variance"], study_dir / "fd_bias_variance.html")
    metrics = {
        "convergence": study,
        "bias_variance": dataset_result["metrics"]["fd_bias_variance"],
        "performance": performance,
        "references": study_references,
        "reference_errors": study_reference_errors,
        "reference_choices": study_reference_choices,
        "study_n_ref": study_n_ref,
        "n_values": n_values,
        "replications": args.convergence_replications,
        "fd_replications": args.fd_replications,
        "fd_paths": args.n_fd,
        "slope_target": -0.5,
    }
    return {
        "params": {
            "approach": "Q2A",
            "study_dataset": args.study_dataset,
            "seed": args.seed,
            "S0": S0,
            "K": K,
            "T": HORIZON,
            "m": MONITORING,
            "barrier": B,
            "r": r,
            "sigma": sigma,
            "n_values": n_values,
            "convergence_replications": args.convergence_replications,
            "fd_replications": args.fd_replications,
            "fd_paths": args.n_fd,
            "fd_bumps": list(FD_BUMPS),
            "study_n_ref": study_n_ref,
        },
        "metrics": metrics,
        "fit_seconds": float(elapsed),
        "history": {
            "timings": {"study_seconds": elapsed},
            "fd_rows": dataset_result["metrics"]["fd_bias_variance"]["rows"],
        },
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    shibor = load_shibor(args.shibor)
    curves = build_rate_curves(shibor)
    sorted_dates = np.array(sorted(curves), dtype="datetime64[ns]")
    if len(sorted_dates) == 0:
        raise ValueError("no usable SHIBOR curves")
    results = {}
    extras = {}
    for offset, code in enumerate(args.datasets, start=1):
        console.rule(f"Q2A {code}")
        result, extra = run_dataset(code, args, shibor, curves, sorted_dates, offset)
        out_dir = args.out_dir / code
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "metrics.json").write_text(
            json.dumps(to_jsonable(result), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        results[code] = result
        extras[code] = extra
    if not args.skip_study and args.study_dataset in results:
        console.rule("Q2A IF study")
        study = run_study(args, results[args.study_dataset], extras[args.study_dataset])
        study_dir = args.out_dir / f"{args.study_dataset}_study"
        (study_dir / "metrics.json").write_text(
            json.dumps(to_jsonable(study), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print_study(study["metrics"]["convergence"], study["metrics"]["bias_variance"], study["metrics"]["performance"])
    console.print("Q2A complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
