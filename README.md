# AP 2026 Q2: Monte Carlo pricing and deep hedging

Q2 of the Advanced Programming 2026 coursework: Monte Carlo pricing of path-dependent payoffs under GBM (Q2A) and CVaR-optimal deep hedging (Q2B).

## Run

`uv sync --python 3.13 && source .venv/bin/activate`

Full reproduction: `./reproduce.sh`
Quick checks (tests + comparison): `./reproduce.sh --quick`
Clean-clone comparison: `python code/check_reproduction.py`

## Results

- `results/approach1/` holds the Q2A runs for IF, CU, RB, SC and the IF study.
- `results/approach2/IF/` holds the Q2B deep-hedging run.

## Provenance

Split from the combined AP 2026 repository at parent tag `submission`, commit `4521da3`. Former `approach3` is now `approach1`; former `approach4` is now `approach2`. See `SPLIT_HISTORY.md` for the commit map and disclosure. `SESSION.md` is the full transcript for all questions and uses the old file names and hashes.
