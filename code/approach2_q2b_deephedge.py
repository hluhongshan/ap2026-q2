#!/usr/bin/env python3
"""Q2 Option B: deep hedging of a European call under GBM.

The policy is a small MLP mapping (S_t/S_0, t/T) to a delta in [0, 1].
It is trained on the brief's P&L definition and evaluated on both the brief
and the self-financing P&L definitions.  All paths are generated with
numpy.default_rng.  torch threads are set explicitly in main().
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import torch
from plotly.subplots import make_subplots
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from scipy.special import ndtr
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
Q2A_PATH = ROOT / "code" / "approach1_q2a_mc.py"
Q2A_METRICS = ROOT / "results" / "approach1" / "IF" / "metrics.json"
DEFAULT_SHIBOR = ROOT / "data" / "clean" / "shibor.csv"
DEFAULT_OUT = ROOT / "results" / "approach2" / "IF"
DEFAULT_LOG = ROOT / ".cache" / "approach2_q2b.log"

KAPPAS_BP = (0, 10, 50)
REBALANCE_DAYS = (1, 5, 21)
MONEYNESS = np.array([0.80, 0.85, 0.90, 0.95, 0.98, 1.00, 1.02, 1.05, 1.10, 1.15, 1.20])
TIME_FRACS = np.array([0.0, 0.25, 0.50, 0.75, 0.95])

console = Console()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Q2 Option B: deep hedging")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset", choices=["IF"], default="IF")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--log-path", type=Path, default=DEFAULT_LOG)
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--n-train", type=int, default=2**17)
    parser.add_argument("--batch-size", type=int, default=2**12)
    parser.add_argument("--n-val", type=int, default=2**17)
    parser.add_argument("--val-seeds", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=5e-3)
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--hidden-layers", type=int, default=2)
    parser.add_argument("--kappas-bp", nargs="+", type=int, default=list(KAPPAS_BP))
    parser.add_argument("--rebalance-days", nargs="+", type=int, default=list(REBALANCE_DAYS))
    parser.add_argument("--T", type=float, default=0.25)
    parser.add_argument("--m", type=int, default=63)
    parser.add_argument("--s0", type=float, default=100.0)
    parser.add_argument("--strike", type=float, default=100.0)
    parser.add_argument("--cvar-level", type=float, default=0.95)
    parser.add_argument("--time-budget-minutes", type=float, default=30.0)
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
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_inside_repo(path: Path) -> Path:
    root = ROOT.resolve()
    resolved = path.resolve()
    if resolved != root and root not in resolved.parents:
        raise ValueError(f"path must stay inside {root}: {resolved}")
    return resolved


def set_torch_threads(threads: int) -> None:
    if threads < 1:
        raise ValueError("--torch-threads must be >= 1")
    torch.set_num_threads(threads)
    try:
        torch.set_num_interop_threads(threads)
    except RuntimeError:
        pass


class RunLog:
    def __init__(self, path: Path) -> None:
        self.path = ensure_inside_repo(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.start = time.perf_counter()

    def write(self, message: str) -> None:
        elapsed = time.perf_counter() - self.start
        line = f"[{elapsed:8.2f}s] {message}"
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
        console.print(line)

    def check_budget(self, limit_minutes: float, phase: str) -> None:
        elapsed = time.perf_counter() - self.start
        if elapsed > limit_minutes * 60.0:
            raise TimeoutError(
                f"runtime guard: {phase} exceeded {limit_minutes:.1f} minutes "
                f"(elapsed {elapsed / 60.0:.1f} min); stop and report before cutting scope"
            )


def load_q2a():
    spec = importlib.util.spec_from_file_location("approach1_q2a_mc", Q2A_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {Q2A_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_market_setup(dataset: str, T: float, shibor_path: Path) -> dict:
    q2a = load_q2a()
    frame = q2a.load_futures(dataset)
    calibration = q2a.calibration_stats(frame["close"].to_numpy(dtype=float), frame["date"])
    sigma = float(calibration["sigma_annualized"])
    shibor = q2a.load_shibor(shibor_path)
    curves = q2a.build_rate_curves(shibor)
    sorted_dates = np.array(sorted(curves), dtype="datetime64[ns]")
    if len(sorted_dates) == 0:
        raise ValueError("no usable SHIBOR curves")
    valuation_date = pd.Timestamp(frame["date"].iloc[-1])
    r, curve_date = q2a.zero_rate(curves, sorted_dates, valuation_date, T)
    reference_sigma = None
    if Q2A_METRICS.exists():
        payload = json.loads(Q2A_METRICS.read_text(encoding="utf-8"))
        reference_sigma = float(payload["params"]["sigma"])
    return {
        "sigma": sigma,
        "sigma_reference": reference_sigma,
        "sigma_matches_reference": bool(
            reference_sigma is None or abs(reference_sigma - sigma) < 1e-12
        ),
        "r": float(r),
        "rate_convention": (
            "SHIBOR zero from approach3: log(1 + quoted*365/360*tau)/tau, linear in tau"
        ),
        "valuation_date": valuation_date.date().isoformat(),
        "shibor_date": curve_date.date().isoformat(),
        "curve_date": curve_date,
        "calibration_window": {
            "start": calibration["window_start"],
            "end": calibration["window_end"],
            "years": calibration["window_years"],
        },
        "input": str((ROOT / "data" / "clean" / f"futures_{dataset}.csv").relative_to(ROOT)),
        "input_sha256": sha256_file(ROOT / "data" / "clean" / f"futures_{dataset}.csv"),
        "shibor": str(shibor_path.relative_to(ROOT)) if shibor_path.is_relative_to(ROOT) else str(shibor_path),
        "shibor_sha256": sha256_file(shibor_path),
    }


def bs_call_price_delta(S, K: float, r: float, sigma: float, tau):
    S_arr = np.asarray(S, dtype=float)
    tau_arr = np.asarray(tau, dtype=float)
    sqrt_tau = np.sqrt(tau_arr)
    d1 = (np.log(S_arr / K) + (r + 0.5 * sigma * sigma) * tau_arr) / (sigma * sqrt_tau)
    d2 = d1 - sigma * sqrt_tau
    price = S_arr * ndtr(d1) - K * np.exp(-r * tau_arr) * ndtr(d2)
    delta = ndtr(d1)
    if np.ndim(price) == 0:
        return float(price), float(delta)
    return price, delta


def gbm_paths(rng: np.random.Generator, S0: float, r: float, sigma: float, T: float, m: int, n: int) -> np.ndarray:
    dt = T / m
    z = rng.standard_normal((n, m))
    log_increments = (r - 0.5 * sigma * sigma) * dt + sigma * math.sqrt(dt) * z
    log_paths = np.log(S0) + np.cumsum(log_increments, axis=1)
    S = np.empty((n, m + 1), dtype=float)
    S[:, 0] = S0
    S[:, 1:] = np.exp(log_paths)
    return S


def pnl_components(S, deltas, r: float, T: float, K: float) -> dict:
    S_arr = np.asarray(S, dtype=float)
    deltas_arr = np.asarray(deltas, dtype=float)
    single = S_arr.ndim == 1
    if single:
        S_arr = S_arr[None, :]
    if deltas_arr.ndim == 1:
        deltas_arr = deltas_arr[None, :]
    m = deltas_arr.shape[-1]
    if S_arr.shape[-1] != m + 1 or deltas_arr.shape[0] != S_arr.shape[0]:
        raise ValueError("expected S with m+1 columns and deltas with m columns")
    dt = T / m
    steps = np.arange(m, dtype=float)
    disc_brief = np.exp(r * (T - steps * dt))
    disc_selffin = np.exp(r * (T - (steps + 1.0) * dt))
    dS = S_arr[:, 1:] - S_arr[:, :-1]
    gains_brief = np.sum(disc_brief * deltas_arr * dS, axis=1)
    gains_selffin = np.sum(
        disc_selffin * deltas_arr * (S_arr[:, 1:] - math.exp(r * dt) * S_arr[:, :-1]), axis=1
    )
    dD = np.diff(np.concatenate([np.zeros((deltas_arr.shape[0], 1)), deltas_arr], axis=1), axis=1)
    cost_brief = np.sum(np.abs(dD) * S_arr[:, :-1], axis=1)
    cost_selffin = np.sum(disc_brief * np.abs(dD) * S_arr[:, :-1], axis=1)
    payoff = np.maximum(S_arr[:, -1] - K, 0.0)
    turnover = np.sum(np.abs(dD) * S_arr[:, :-1], axis=1) / S_arr[:, 0]
    out = {
        "gains_brief": gains_brief,
        "gains_selffin": gains_selffin,
        "cost_brief": cost_brief,
        "cost_selffin": cost_selffin,
        "payoff": payoff,
        "turnover": turnover,
    }
    if single:
        return {key: float(val[0]) for key, val in out.items()}
    return out


def pnl_brief(S, deltas, kappa: float, r: float, T: float, K: float):
    comp = pnl_components(S, deltas, r, T, K)
    value = comp["gains_brief"] - kappa * comp["cost_brief"] - comp["payoff"]
    return value


def pnl_selffin(S, deltas, kappa: float, r: float, T: float, K: float):
    comp = pnl_components(S, deltas, r, T, K)
    value = comp["gains_selffin"] - kappa * comp["cost_selffin"] - comp["payoff"]
    return value


def pnl_brief_torch(S: torch.Tensor, deltas: torch.Tensor, kappa: float, r: float, T: float, K: float) -> torch.Tensor:
    m = deltas.shape[1]
    dt = T / m
    steps = torch.arange(m, dtype=S.dtype, device=S.device)
    disc = torch.exp(r * (T - steps * dt))
    dS = S[:, 1:] - S[:, :-1]
    gains = (disc * deltas * dS).sum(dim=1)
    zeros = torch.zeros((deltas.shape[0], 1), dtype=S.dtype, device=S.device)
    dD = torch.diff(torch.cat([zeros, deltas], dim=1), dim=1)
    cost = (torch.abs(dD) * S[:, :-1]).sum(dim=1)
    payoff = torch.clamp(S[:, -1] - K, min=0.0)
    return gains - kappa * cost - payoff


def empirical_cvar(losses, level: float = 0.95) -> float:
    values = np.asarray(losses, dtype=float).ravel()
    n = values.size
    if n == 0:
        raise ValueError("empty loss sample")
    tail = 1.0 - level
    k = max(1, int(math.ceil(tail * n - 1e-12)))
    k = min(k, n)
    return float(np.mean(np.partition(values, n - k)[n - k:]))


def empirical_cvar_torch(losses: torch.Tensor, level: float = 0.95) -> torch.Tensor:
    n = losses.numel()
    tail = 1.0 - level
    k = max(1, int(math.ceil(tail * n - 1e-12)))
    k = min(k, n)
    return torch.topk(losses, k, largest=True, sorted=False).values.mean()


def bs_delta_path(S: np.ndarray, K: float, r: float, sigma: float, T: float, m: int, rebalance_days: int) -> np.ndarray:
    n = S.shape[0]
    dt = T / m
    deltas = np.empty((n, m), dtype=float)
    last = None
    for t in range(m):
        if t % rebalance_days == 0:
            tau = T - t * dt
            if tau <= 0.0:
                last = (S[:, t] > K).astype(float)
            else:
                _, last = bs_call_price_delta(S[:, t], K, r, sigma, tau)
        deltas[:, t] = last
    return deltas


class MLPPolicy(nn.Module):
    def __init__(self, hidden: int = 48, hidden_layers: int = 2) -> None:
        super().__init__()
        if hidden_layers < 1 or hidden_layers > 3:
            raise ValueError("hidden_layers must be between 1 and 3")
        layers: list[nn.Module] = []
        in_dim = 2
        for _ in range(hidden_layers):
            layers.append(nn.Linear(in_dim, hidden))
            layers.append(nn.Tanh())
            in_dim = hidden
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)
        self.hidden_layers = hidden_layers

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(x)).squeeze(-1)


def build_base_state(seed: int, hidden: int, hidden_layers: int) -> dict:
    torch.manual_seed(seed)
    policy = MLPPolicy(hidden, hidden_layers)
    with torch.no_grad():
        policy.net[-1].bias.fill_(0.0)
    return {key: val.detach().clone() for key, val in policy.state_dict().items()}


def make_constant_policy(delta_value: float, hidden: int = 8, hidden_layers: int = 1) -> MLPPolicy:
    if not 0.0 < delta_value < 1.0:
        raise ValueError("delta_value must lie strictly inside (0, 1)")
    policy = MLPPolicy(hidden, hidden_layers)
    with torch.no_grad():
        for param in policy.parameters():
            param.zero_()
        policy.net[-1].bias.fill_(math.log(delta_value / (1.0 - delta_value)))
    return policy


def policy_deltas(policy: MLPPolicy, S: np.ndarray, S0: float, T: float, m: int, batch: int = 16384) -> np.ndarray:
    policy.eval()
    n = S.shape[0]
    time_grid = torch.arange(m, dtype=torch.float32) / m
    out = np.empty((n, m), dtype=float)
    with torch.no_grad():
        for start in range(0, n, batch):
            stop = min(start + batch, n)
            S_b = torch.from_numpy(S[start:stop]).float()
            x = torch.stack(
                [S_b[:, :m] / S0, time_grid.unsqueeze(0).expand(stop - start, m)], dim=-1
            )
            out[start:stop] = policy(x).numpy()
    return out


def train_policy(
    S_train: torch.Tensor,
    time_grid: torch.Tensor,
    kappa: float,
    base_state: dict,
    args: argparse.Namespace,
    log: RunLog,
    tag: str,
) -> tuple[MLPPolicy, dict, float]:
    policy = MLPPolicy(args.hidden, args.hidden_layers)
    policy.load_state_dict(base_state)
    optimizer = torch.optim.Adam(policy.parameters(), lr=args.lr)
    n = S_train.shape[0]
    m = args.m
    history = {"epochs": [], "train_cvar": [], "epoch_seconds": []}
    start_time = time.perf_counter()
    for epoch in range(args.epochs):
        epoch_start = time.perf_counter()
        perm = np.random.default_rng([args.seed, 301, epoch]).permutation(n)
        batch_losses = []
        for start in range(0, n, args.batch_size):
            stop = min(start + args.batch_size, n)
            idx = torch.from_numpy(perm[start:stop])
            S_b = S_train[idx]
            x = torch.stack(
                [S_b[:, :m] / args.s0, time_grid.unsqueeze(0).expand(stop - start, m)], dim=-1
            )
            deltas = policy(x)
            value = pnl_brief_torch(S_b, deltas, kappa, args.r, args.T, args.strike)
            loss = empirical_cvar_torch(-value, args.cvar_level)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            batch_losses.append(float(loss.detach()))
        epoch_cvar = float(np.mean(batch_losses))
        epoch_seconds = time.perf_counter() - epoch_start
        history["epochs"].append(epoch + 1)
        history["train_cvar"].append(epoch_cvar)
        history["epoch_seconds"].append(epoch_seconds)
        log.write(
            f"{tag} epoch {epoch + 1:4d}/{args.epochs} train_CVaR={epoch_cvar:.6f} "
            f"({epoch_seconds:.2f}s)"
        )
        log.check_budget(args.time_budget_minutes, f"{tag} training")
    seconds = time.perf_counter() - start_time
    return policy, history, seconds


def distribution_metrics(pnl: np.ndarray, turnover: np.ndarray, premium_fv: float, level: float) -> dict:
    cvar = empirical_cvar(-pnl, level)
    return {
        "cvar": cvar,
        "net_cvar": cvar - premium_fv,
        "variance": float(np.var(pnl, ddof=1)),
        "mean_pnl": float(np.mean(pnl)),
        "turnover": float(np.mean(turnover)),
    }


def metrics_for_kappa(comp: dict, kappa: float, premium_fv: float, level: float) -> dict:
    brief_pnl = comp["gains_brief"] - kappa * comp["cost_brief"] - comp["payoff"]
    selffin_pnl = comp["gains_selffin"] - kappa * comp["cost_selffin"] - comp["payoff"]
    return {
        "brief": distribution_metrics(brief_pnl, comp["turnover"], premium_fv, level),
        "selffin": distribution_metrics(selffin_pnl, comp["turnover"], premium_fv, level),
        "gap_brief_minus_selffin": {
            "cvar": empirical_cvar(-brief_pnl, level) - empirical_cvar(-selffin_pnl, level),
            "net_cvar": (empirical_cvar(-brief_pnl, level) - premium_fv)
            - (empirical_cvar(-selffin_pnl, level) - premium_fv),
        },
    }


def aggregate_records(records: list[dict]) -> dict:
    out = {}
    for key in records[0]:
        values = np.array([record[key] for record in records], dtype=float)
        out[f"{key}_mean"] = float(np.mean(values))
        out[f"{key}_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
    return out


def aggregate_cell(per_seed: list[dict]) -> dict:
    out = {}
    for pnl_name in ("brief", "selffin"):
        out[pnl_name] = aggregate_records([record[pnl_name] for record in per_seed])
    out["gap_brief_minus_selffin"] = aggregate_records(
        [record["gap_brief_minus_selffin"] for record in per_seed]
    )
    return out


def sensitivity_grid(policy: MLPPolicy, S0: float, K: float, r: float, sigma: float, T: float) -> dict:
    rows = []
    for frac in TIME_FRACS:
        tau = T * (1.0 - frac)
        S_grid = S0 * MONEYNESS
        if tau <= 0.0:
            bs_delta = (S_grid > K).astype(float)
        else:
            _, bs_delta = bs_call_price_delta(S_grid, K, r, sigma, tau)
        x = torch.tensor(
            np.column_stack([MONEYNESS, np.full_like(MONEYNESS, frac)]), dtype=torch.float32
        )
        with torch.no_grad():
            policy_delta = policy(x).numpy()
        rows.append(
            {
                "time_fraction": float(frac),
                "tau": float(tau),
                "moneyness": MONEYNESS.tolist(),
                "policy_delta": policy_delta.tolist(),
                "bs_delta": bs_delta.tolist(),
                "mean_abs_diff": float(np.mean(np.abs(policy_delta - bs_delta))),
                "max_abs_diff": float(np.max(np.abs(policy_delta - bs_delta))),
                "mean_policy_delta": float(np.mean(policy_delta)),
                "std_policy_delta": float(np.std(policy_delta, ddof=1)),
                "mean_bs_delta": float(np.mean(bs_delta)),
            }
        )
    all_policy = np.concatenate([np.array(row["policy_delta"]) for row in rows])
    all_bs = np.concatenate([np.array(row["bs_delta"]) for row in rows])
    return {
        "rows": rows,
        "mean_abs_diff": float(np.mean(np.abs(all_policy - all_bs))),
        "max_abs_diff": float(np.max(np.abs(all_policy - all_bs))),
        "mean_policy_delta": float(np.mean(all_policy)),
        "std_policy_delta": float(np.std(all_policy, ddof=1)),
        "mean_bs_delta": float(np.mean(all_bs)),
    }


def training_figure(histories: dict, path: Path) -> None:
    fig = go.Figure()
    for kappa_bp, history in histories.items():
        fig.add_trace(
            go.Scatter(
                x=history["epochs"],
                y=history["train_cvar"],
                mode="lines",
                name=f"kappa={kappa_bp}bp",
            )
        )
    fig.update_layout(
        title="Q2B training curves: empirical CVaR_0.95 of -V_T (brief P&L)",
        xaxis_title="epoch",
        yaxis_title="training CVaR",
        height=600,
        width=1000,
    )
    fig.write_html(path, include_plotlyjs=True)


def policy_figure(sensitivities: dict, path: Path) -> None:
    kappas = list(sensitivities)
    fig = make_subplots(
        rows=len(kappas),
        cols=len(TIME_FRACS),
        subplot_titles=tuple(
            f"t/T={frac:.2f}, kappa={kappa_bp}bp" for kappa_bp in kappas for frac in TIME_FRACS
        ),
        shared_xaxes=True,
        shared_yaxes=True,
    )
    for row, kappa_bp in enumerate(kappas, start=1):
        for col, item in enumerate(sensitivities[kappa_bp]["rows"], start=1):
            fig.add_trace(
                go.Scatter(
                    x=item["moneyness"],
                    y=item["policy_delta"],
                    mode="lines+markers",
                    name="deep policy",
                    legendgroup="deep",
                    showlegend=row == 1 and col == 1,
                    line={"color": "#1f77b4"},
                ),
                row=row,
                col=col,
            )
            fig.add_trace(
                go.Scatter(
                    x=item["moneyness"],
                    y=item["bs_delta"],
                    mode="lines",
                    name="Black-Scholes",
                    legendgroup="bs",
                    showlegend=row == 1 and col == 1,
                    line={"color": "#ff7f0e", "dash": "dash"},
                ),
                row=row,
                col=col,
            )
            fig.update_xaxes(title_text="S/S0", row=row, col=col)
            fig.update_yaxes(title_text="delta", row=row, col=col)
    fig.update_layout(
        title="Q2B learned hedge ratio versus Black-Scholes delta",
        height=260 * len(kappas),
        width=1600,
    )
    fig.write_html(path, include_plotlyjs=True)


def print_setup(setup: dict, args: argparse.Namespace, premium: float, premium_fv: float) -> None:
    panel = Panel.fit(
        f"dataset={args.dataset}  sigma={setup['sigma']:.12f}  r={setup['r']:.12f}\n"
        f"S0={args.s0:.2f}  K={args.strike:.2f}  T={args.T:.4f}  m={args.m}  "
        f"dt={args.T / args.m:.8f}\n"
        f"BS premium={premium:.8f}  future value={premium_fv:.8f}  "
        f"valuation={setup['valuation_date']}",
        title="Q2B setup",
    )
    console.print(panel)


def print_table(rows: list[dict]) -> None:
    table = Table(title="Q2B deep hedging and Black-Scholes benchmarks")
    table.add_column("Method")
    table.add_column("Rebal", justify="right")
    table.add_column("Policy k", justify="right")
    table.add_column("Eval k", justify="right")
    table.add_column("P&L")
    table.add_column("CVaR", justify="right")
    table.add_column("Net CVaR", justify="right")
    table.add_column("Var", justify="right")
    table.add_column("E[P&L]", justify="right")
    table.add_column("Turnover", justify="right")
    for row in rows:
        table.add_row(
            row["method"],
            "-" if row["rebalance_days"] is None else str(row["rebalance_days"]),
            "-" if row["policy_kappa_bp"] is None else str(row["policy_kappa_bp"]),
            str(row["eval_kappa_bp"]),
            row["pnl"],
            f"{row['cvar_mean']:.5f} ± {row['cvar_std']:.5f}",
            f"{row['net_cvar_mean']:.5f} ± {row['net_cvar_std']:.5f}",
            f"{row['variance_mean']:.5f} ± {row['variance_std']:.5f}",
            f"{row['mean_pnl_mean']:.5f} ± {row['mean_pnl_std']:.5f}",
            f"{row['turnover_mean']:.5f} ± {row['turnover_std']:.5f}",
        )
    console.print(table)


def print_cross(cross_rows: list[dict]) -> None:
    table = Table(title="Q2B cross-kappa evaluation")
    table.add_column("Policy k", justify="right")
    table.add_column("Eval k", justify="right")
    table.add_column("CVaR brief", justify="right")
    table.add_column("Net CVaR brief", justify="right")
    table.add_column("CVaR selffin", justify="right")
    table.add_column("Net CVaR selffin", justify="right")
    table.add_column("Turnover", justify="right")
    for row in cross_rows:
        table.add_row(
            str(row["policy_kappa_bp"]),
            str(row["eval_kappa_bp"]),
            f"{row['cvar_brief_mean']:.5f} ± {row['cvar_brief_std']:.5f}",
            f"{row['net_cvar_brief_mean']:.5f} ± {row['net_cvar_brief_std']:.5f}",
            f"{row['cvar_selffin_mean']:.5f} ± {row['cvar_selffin_std']:.5f}",
            f"{row['net_cvar_selffin_mean']:.5f} ± {row['net_cvar_selffin_std']:.5f}",
            f"{row['turnover_mean']:.5f} ± {row['turnover_std']:.5f}",
        )
    console.print(table)


def print_sensitivity(sensitivities: dict) -> None:
    table = Table(title="Q2B policy sensitivity on the moneyness/time grid")
    table.add_column("Policy k", justify="right")
    table.add_column("Mean |policy-BS|", justify="right")
    table.add_column("Max |policy-BS|", justify="right")
    table.add_column("Mean policy delta", justify="right")
    table.add_column("Std policy delta", justify="right")
    table.add_column("Mean BS delta", justify="right")
    for kappa_bp, item in sensitivities.items():
        table.add_row(
            str(kappa_bp),
            f"{item['mean_abs_diff']:.5f}",
            f"{item['max_abs_diff']:.5f}",
            f"{item['mean_policy_delta']:.5f}",
            f"{item['std_policy_delta']:.5f}",
            f"{item['mean_bs_delta']:.5f}",
        )
    console.print(table)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    set_torch_threads(args.torch_threads)
    out_dir = ensure_inside_repo(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log = RunLog(args.log_path)
    log.write(f"start args={vars(args)}")
    log.write(f"torch threads={torch.get_num_threads()} interop={torch.get_num_interop_threads()}")

    setup = load_market_setup(args.dataset, args.T, DEFAULT_SHIBOR)
    args.sigma = setup["sigma"]
    args.r = setup["r"]
    if not setup["sigma_matches_reference"]:
        raise RuntimeError(
            f"sigma {setup['sigma']} does not match approach3 reference {setup['sigma_reference']}"
        )
    premium, _ = bs_call_price_delta(args.s0, args.strike, args.r, args.sigma, args.T)
    premium_fv = premium * math.exp(args.r * args.T)
    print_setup(setup, args, premium, premium_fv)
    log.write(
        f"sigma={args.sigma:.12f} r={args.r:.12f} premium={premium:.8f} premium_fv={premium_fv:.8f}"
    )

    train_rng = np.random.default_rng([args.seed, 101])
    S_train_np = gbm_paths(train_rng, args.s0, args.r, args.sigma, args.T, args.m, args.n_train)
    S_train = torch.from_numpy(S_train_np).float()
    time_grid = torch.arange(args.m, dtype=torch.float32) / args.m
    log.write(f"generated {args.n_train} training paths with seed stream [{args.seed}, 101]")

    base_state = build_base_state(args.seed, args.hidden, args.hidden_layers)
    policies = {}
    histories = {}
    fit_seconds = {}
    for kappa_bp in args.kappas_bp:
        kappa = kappa_bp / 10_000.0
        tag = f"kappa={kappa_bp}bp"
        log.write(f"training policy {tag}")
        policy, history, seconds = train_policy(
            S_train, time_grid, kappa, base_state, args, log, tag
        )
        policies[kappa_bp] = policy
        histories[kappa_bp] = history
        fit_seconds[kappa_bp] = seconds
        log.write(f"finished policy {tag} in {seconds:.2f}s")
        log.check_budget(args.time_budget_minutes, tag)
    del S_train, S_train_np

    per_seed_bs: dict[tuple[int, int], list[dict]] = {}
    per_seed_deep: dict[tuple[int, int], list[dict]] = {}
    for seed_index in range(args.val_seeds):
        val_seed = [args.seed, 5001 + seed_index]
        log.write(f"validation seed {seed_index + 1}/{args.val_seeds}: stream {val_seed}")
        rng = np.random.default_rng(val_seed)
        S_val = gbm_paths(rng, args.s0, args.r, args.sigma, args.T, args.m, args.n_val)
        for rebalance_days in args.rebalance_days:
            deltas = bs_delta_path(
                S_val, args.strike, args.r, args.sigma, args.T, args.m, rebalance_days
            )
            comp = pnl_components(S_val, deltas, args.r, args.T, args.strike)
            for kappa_bp in args.kappas_bp:
                key = (rebalance_days, kappa_bp)
                per_seed_bs.setdefault(key, []).append(
                    metrics_for_kappa(comp, kappa_bp / 10_000.0, premium_fv, args.cvar_level)
                )
        for policy_kappa_bp, policy in policies.items():
            deltas = policy_deltas(policy, S_val, args.s0, args.T, args.m)
            comp = pnl_components(S_val, deltas, args.r, args.T, args.strike)
            for eval_kappa_bp in args.kappas_bp:
                key = (policy_kappa_bp, eval_kappa_bp)
                per_seed_deep.setdefault(key, []).append(
                    metrics_for_kappa(
                        comp, eval_kappa_bp / 10_000.0, premium_fv, args.cvar_level
                    )
                )
        log.check_budget(args.time_budget_minutes, f"validation seed {seed_index + 1}")

    table_rows = []
    for (rebalance_days, kappa_bp), records in per_seed_bs.items():
        cell = aggregate_cell(records)
        for pnl_name in ("brief", "selffin"):
            row = {
                "method": "BS",
                "rebalance_days": int(rebalance_days),
                "policy_kappa_bp": None,
                "eval_kappa_bp": int(kappa_bp),
                "pnl": pnl_name,
            }
            for key, value in cell[pnl_name].items():
                row[key] = value
            for key, value in cell["gap_brief_minus_selffin"].items():
                row[f"gap_{key}"] = value
            table_rows.append(row)
    for (policy_kappa_bp, eval_kappa_bp), records in per_seed_deep.items():
        if policy_kappa_bp != eval_kappa_bp:
            continue
        cell = aggregate_cell(records)
        for pnl_name in ("brief", "selffin"):
            row = {
                "method": "deep",
                "rebalance_days": None,
                "policy_kappa_bp": int(policy_kappa_bp),
                "eval_kappa_bp": int(eval_kappa_bp),
                "pnl": pnl_name,
            }
            for key, value in cell[pnl_name].items():
                row[key] = value
            for key, value in cell["gap_brief_minus_selffin"].items():
                row[f"gap_{key}"] = value
            table_rows.append(row)

    cross_rows = []
    for (policy_kappa_bp, eval_kappa_bp), records in per_seed_deep.items():
        cell = aggregate_cell(records)
        cross_rows.append(
            {
                "policy_kappa_bp": int(policy_kappa_bp),
                "eval_kappa_bp": int(eval_kappa_bp),
                "cvar_brief_mean": cell["brief"]["cvar_mean"],
                "cvar_brief_std": cell["brief"]["cvar_std"],
                "net_cvar_brief_mean": cell["brief"]["net_cvar_mean"],
                "net_cvar_brief_std": cell["brief"]["net_cvar_std"],
                "cvar_selffin_mean": cell["selffin"]["cvar_mean"],
                "cvar_selffin_std": cell["selffin"]["cvar_std"],
                "net_cvar_selffin_mean": cell["selffin"]["net_cvar_mean"],
                "net_cvar_selffin_std": cell["selffin"]["net_cvar_std"],
                "turnover_mean": cell["brief"]["turnover_mean"],
                "turnover_std": cell["brief"]["turnover_std"],
            }
        )
    cross_rows.sort(key=lambda row: (row["policy_kappa_bp"], row["eval_kappa_bp"]))

    sensitivities = {
        kappa_bp: sensitivity_grid(
            policy, args.s0, args.strike, args.r, args.sigma, args.T
        )
        for kappa_bp, policy in policies.items()
    }
    sensitivity_summary = {}
    for kappa_bp in args.kappas_bp:
        deep_row = next(
            row
            for row in table_rows
            if row["method"] == "deep"
            and row["policy_kappa_bp"] == kappa_bp
            and row["pnl"] == "selffin"
        )
        sensitivity_summary[str(kappa_bp)] = {
            "turnover_mean": deep_row["turnover_mean"],
            "turnover_std": deep_row["turnover_std"],
            "mean_abs_diff_vs_bs": sensitivities[kappa_bp]["mean_abs_diff"],
            "max_abs_diff_vs_bs": sensitivities[kappa_bp]["max_abs_diff"],
            "mean_policy_delta": sensitivities[kappa_bp]["mean_policy_delta"],
            "std_policy_delta": sensitivities[kappa_bp]["std_policy_delta"],
            "mean_bs_delta": sensitivities[kappa_bp]["mean_bs_delta"],
        }

    bs_daily = next(
        row
        for row in table_rows
        if row["method"] == "BS"
        and row["rebalance_days"] == 1
        and row["eval_kappa_bp"] == 0
        and row["pnl"] == "selffin"
    )
    deep_zero = next(
        row
        for row in table_rows
        if row["method"] == "deep"
        and row["policy_kappa_bp"] == 0
        and row["pnl"] == "selffin"
    )
    bs_daily_brief = next(
        row
        for row in table_rows
        if row["method"] == "BS"
        and row["rebalance_days"] == 1
        and row["eval_kappa_bp"] == 0
        and row["pnl"] == "brief"
    )
    deep_zero_brief = next(
        row
        for row in table_rows
        if row["method"] == "deep"
        and row["policy_kappa_bp"] == 0
        and row["pnl"] == "brief"
    )
    ratio = deep_zero["cvar_mean"] / bs_daily["cvar_mean"] if bs_daily["cvar_mean"] > 0 else None
    ratio_brief = (
        deep_zero_brief["cvar_mean"] / bs_daily_brief["cvar_mean"]
        if bs_daily_brief["cvar_mean"] > 0
        else None
    )
    deep_zero_check = {
        "selffin": {
            "deep_cvar_kappa0": deep_zero["cvar_mean"],
            "bs_daily_cvar_kappa0": bs_daily["cvar_mean"],
            "ratio": ratio,
        },
        "brief": {
            "deep_cvar_kappa0": deep_zero_brief["cvar_mean"],
            "bs_daily_cvar_kappa0": bs_daily_brief["cvar_mean"],
            "ratio": ratio_brief,
        },
        "close_to_bs_daily": bool(
            ratio is not None
            and ratio <= 1.25
            and (ratio_brief is None or ratio_brief <= 1.25)
        ),
        "note": (
            "If a ratio is clearly above one, this is treated as a training failure, "
            "not as evidence about the method."
        ),
    }

    table_frame = pd.DataFrame(table_rows)
    table_frame.to_csv(out_dir / "table.csv", index=False)
    cross_frame = pd.DataFrame(cross_rows)
    cross_frame.to_csv(out_dir / "cross_kappa.csv", index=False)
    training_figure(histories, out_dir / "training.html")
    policy_figure(sensitivities, out_dir / "policy.html")

    gap_values = [row["gap_cvar_mean"] for row in table_rows if row["pnl"] == "brief"]
    brief_gap = {
        "mean_cvar_gap_across_cells": float(np.mean(gap_values)),
        "by_cell": [
            {
                "method": row["method"],
                "rebalance_days": row["rebalance_days"],
                "policy_kappa_bp": row["policy_kappa_bp"],
                "eval_kappa_bp": row["eval_kappa_bp"],
                "pnl": row["pnl"],
                "cvar_gap_mean": row["gap_cvar_mean"],
                "cvar_gap_std": row["gap_cvar_std"],
                "net_cvar_gap_mean": row["gap_net_cvar_mean"],
                "net_cvar_gap_std": row["gap_net_cvar_std"],
            }
            for row in table_rows
            if row["pnl"] == "brief"
        ]
    }

    params = {
        "approach": "Q2B",
        "dataset": args.dataset,
        "seed": args.seed,
        "torch_threads": args.torch_threads,
        "input": setup["input"],
        "input_sha256": setup["input_sha256"],
        "shibor": setup["shibor"],
        "shibor_sha256": setup["shibor_sha256"],
        "valuation_date": setup["valuation_date"],
        "shibor_date": setup["shibor_date"],
        "S0": args.s0,
        "K": args.strike,
        "T": args.T,
        "m": args.m,
        "dt": args.T / args.m,
        "sigma": args.sigma,
        "sigma_source": "approach3 5-year calibration of IF futures close",
        "sigma_reference_approach3": setup["sigma_reference"],
        "sigma_matches_reference": setup["sigma_matches_reference"],
        "r": args.r,
        "rate_convention": setup["rate_convention"],
        "calibration_window": setup["calibration_window"],
        "premium_bs": premium,
        "premium_fv": premium_fv,
        "kappas_bp": args.kappas_bp,
        "rebalance_days": args.rebalance_days,
        "cvar_level": args.cvar_level,
        "cvar_tail_fraction": 1.0 - args.cvar_level,
        "cvar_definition": (
            "mean of the largest ceil((1-level)*n) losses; losses are -V_T"
        ),
        "n_train_paths": args.n_train,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "lr": args.lr,
        "hidden": args.hidden,
        "hidden_layers": args.hidden_layers,
        "dtype": "float32",
        "n_val_paths": args.n_val,
        "val_seeds": args.val_seeds,
        "training_path_seed": [args.seed, 101],
        "validation_path_seeds": [[args.seed, 5001 + i] for i in range(args.val_seeds)],
        "policy_inputs": ["S_t/S0", "t/T"],
        "policy_output": "sigmoid in [0,1]",
        "pnl_definitions": {
            "brief": (
                "sum_t exp(r*(T-t*dt)) D_t (S_{t+1}-S_t) "
                "- kappa sum_t |D_t-D_{t-1}| S_t - payoff, D_{-1}=0, no terminal unwind"
            ),
            "selffin": (
                "sum_t exp(r*(T-(t+1)*dt)) D_t (S_{t+1}-exp(r*dt) S_t) "
                "- kappa sum_t exp(r*(T-t*dt)) |D_t-D_{t-1}| S_t - payoff, D_{-1}=0, no terminal unwind"
            ),
            "training_pnl": "brief",
        },
        "net_cvar_definition": "CVaR - premium_bs * exp(r*T)",
        "turnover_definition": "mean over paths of sum_t |D_t-D_{t-1}| S_t / S0",
        "limitation": (
            "The policy sees only (S_t/S0, t/T), not the previous position, so it cannot "
            "represent a no-trade region. This is a structural limitation of the specified input set."
        ),
    }

    metrics = {
        "setup": {
            "sigma": args.sigma,
            "r": args.r,
            "premium_bs": premium,
            "premium_fv": premium_fv,
            "valuation_date": setup["valuation_date"],
            "calibration_window": setup["calibration_window"],
            "S0": args.s0,
            "K": args.strike,
            "T": args.T,
            "m": args.m,
        },
        "table": table_rows,
        "cross_kappa": cross_rows,
        "sensitivity": sensitivities,
        "sensitivity_summary": sensitivity_summary,
        "brief_vs_selffin_gap": brief_gap,
        "deep_kappa0_vs_bs_daily": deep_zero_check,
        "training_summary": {
            str(kappa_bp): {
                "initial_train_cvar": history["train_cvar"][0],
                "final_train_cvar": history["train_cvar"][-1],
                "min_train_cvar": float(np.min(history["train_cvar"])),
                "seconds": fit_seconds[kappa_bp],
            }
            for kappa_bp, history in histories.items()
        },
        "limitation": params["limitation"],
    }

    history_payload = {
        "training": {str(kappa_bp): value for kappa_bp, value in histories.items()},
        "timings": {
            "fit_seconds": fit_seconds,
            "total_fit_seconds": float(sum(fit_seconds.values())),
            "wall_seconds": float(time.perf_counter() - log.start),
        },
        "config": {
            "seed": args.seed,
            "torch_threads": args.torch_threads,
            "n_train": args.n_train,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "lr": args.lr,
            "hidden": args.hidden,
            "hidden_layers": args.hidden_layers,
            "dtype": "float32",
        },
    }

    result = {
        "params": params,
        "metrics": metrics,
        "fit_seconds": float(sum(fit_seconds.values())),
        "history": history_payload,
    }
    (out_dir / "metrics.json").write_text(
        json.dumps(to_jsonable(result), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    log.write(f"wrote {out_dir / 'metrics.json'}")
    print_table(table_rows)
    print_cross(cross_rows)
    print_sensitivity(sensitivities)
    console.print(
        f"Deep kappa=0 selffin CVaR / BS daily selffin CVaR = {ratio:.4f} "
        f"(close_to_bs_daily={deep_zero_check['close_to_bs_daily']})"
    )
    console.print(f"Total fit seconds: {result['fit_seconds']:.2f}")
    log.write("Q2B complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
