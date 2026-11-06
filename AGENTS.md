# AGENTS.md

## Goal
Q2 of the Advanced Programming 2026 brief, options A and B: price path-dependent payoffs by Monte Carlo under GBM, and learn a hedge minimizing CVaR_0.95 of -V_T. Option A is the Monte Carlo engine; Option B is deep hedging.

## Environment
Python 3.13, managed with uv, dependencies pinned.
uv sync --python 3.13 && source .venv/bin/activate
./reproduce.sh --quick   # tests + compare only
./reproduce.sh           # offline data check, tests, Q2 approaches, compare
python code/check_reproduction.py

## Approach map
- approach1_q2a_mc.py: Q2A Monte Carlo pricing (formerly approach3)
- approach2_q2b_deephedge.py: Q2B deep hedging (formerly approach4)

## Data
data/clean/futures_IF.csv, futures_RB.csv, futures_CU.csv, futures_SC.csv and shibor.csv are committed snapshots. data/metadata.json is filtered to Q2 artifacts; data/fetch.py --offline verifies every checksum. Raw futures downloads are gitignored and are not part of this repository.

## Conventions
Seed: --seed 42 by default, CLI-overridable; every stochastic step is seeded.
Each code/approach*.py: argparse CLI, load -> verify -> split -> fit -> evaluate -> write.
Output: results/approachN/<dataset>/metrics.json = {params, metrics, fit_seconds, history}.
Terminal output via rich; figures via plotly HTML with include_plotlyjs=True.
Q2B reads the Q2A IF metrics for the sigma reference check; run Q2A first.

## Reproduction
`./reproduce.sh` is the full offline path; `./reproduce.sh --quick` runs tests and compare only.
`python code/check_reproduction.py` compares `.cache/clean_clone_full/results` against the committed `results`, ignores timing fields, and uses relative tolerance 1e-6.

## Provenance
This repository was split from the combined AP 2026 repository at parent tag `submission`, commit `4521da3`. See `SPLIT_HISTORY.md` for the old-to-new commit map, renamed paths and the split disclosure.

## Agent rules
Write nothing outside the repo root; tool caches stay in the repo and are gitignored.
Propose first and wait for approval before any destructive operation.
Commit after each milestone (data:, feat:, exp:, refactor:, docs:).
Never commit credentials. Never invent results: every reported number comes from a metrics.json.
