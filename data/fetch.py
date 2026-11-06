from __future__ import annotations


import argparse


import hashlib


import json


import logging


import re


import signal


import sys


import time


from contextlib import contextmanager


from datetime import date, datetime, timedelta


from pathlib import Path


import pandas as pd


import requests


from rich.console import Console


from rich.table import Table


ROOT = Path(__file__).resolve().parents[1]


def rel_path(path: Path | str | None) -> str | None:
    if path is None:
        return None
    candidate = Path(path)
    try:
        return str(candidate.resolve().relative_to(ROOT))
    except ValueError:
        return str(candidate)


RAW = ROOT / "data" / "raw"


CLEAN = ROOT / "data" / "clean"


CFFEX_DIR = RAW / "cffex"


Q3_DIR = RAW / "q3"


CACHE = ROOT / ".cache"


METADATA = ROOT / "data" / "metadata.json"


MACRO_PANEL = CLEAN / "macro_panel.csv"


MACRO_META = CLEAN / "macro_meta.csv"


SHIBOR_CLEAN = CLEAN / "shibor.csv"


SAMPLE_MONTHS = [
    "2024-03",
    "2024-09",
    "2025-03",
    "2025-09",
    "2026-03",
    "2026-09",
]


CFFEX_URL = "http://www.cffex.com.cn/sj/hqsj/rtj/{ym}/{dd}/{ymd}_1.csv"


USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"
)


OPTION_COLUMNS = [
    "date",
    "contract",
    "expiry",
    "cp",
    "K",
    "open",
    "high",
    "low",
    "close",
    "settle",
    "volume",
    "oi",
    "S",
]


SHIBOR_COLUMNS = ["date", "ON", "1W", "2W", "1M", "3M", "6M", "9M", "1Y"]


SINA_COLUMNS = [
    "date",
    "contract",
    "sina_symbol",
    "cffex_close",
    "sina_close",
    "diff",
    "status",
    "error",
]


FUTURES_SYMBOLS = {"IF": "IF0", "RB": "RB0", "CU": "CU0", "SC": "SC0"}


FUTURES_COLUMNS = ["date", "open", "high", "low", "close", "volume", "oi"]


log = logging.getLogger("q1-fetch")


console = Console()


INTERFACES: list[dict] = []


class CallTimeout(Exception):
    pass


def _timeout_handler(signum, frame):
    raise CallTimeout("call exceeded its wall-clock limit")


@contextmanager
def time_limit(seconds: int):
    previous = signal.signal(signal.SIGALRM, _timeout_handler)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def record(name: str, status: str, detail: str = "", path: str | None = None):
    INTERFACES.append(
        {
            "interface": name,
            "status": status,
            "detail": detail,
            "path": rel_path(path),
            "retrieved_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        }
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch_ak_csv(name: str, path: Path, call, attempts: int = 1, delay: float = 0.0, **kwargs):
    if path.exists() and path.stat().st_size > 0:
        record(name, "skipped", f"cached {path.name}", str(path))
        log.info("%s skipped; cached", name)
        return
    failures = []
    for attempt in range(1, attempts + 1):
        try:
            with time_limit(180):
                frame = call(**kwargs)
            if frame is None or len(frame) == 0:
                raise ValueError("empty response")
            frame.to_csv(path, index=False, encoding="utf-8")
            record(name, "ok", f"rows={len(frame)}", str(path))
            log.info("%s ok (%d rows)", name, len(frame))
            return
        except Exception as exc:
            message = f"{type(exc).__name__}: {exc}"
            failures.append(f"attempt {attempt}: {message}")
            log.error("%s attempt %d failed: %s", name, attempt, message)
            if attempt < attempts and delay:
                time.sleep(delay)
    record(name, "failed", " | ".join(failures), str(path))


Q3_INTERFACES = [
    "macro_china_pmi",
    "macro_china_non_man_pmi",
    "macro_china_cx_pmi_yearly",
    "macro_china_cx_services_pmi_yearly",
    "macro_china_cpi",
    "macro_china_cpi_yearly",
    "macro_china_ppi",
    "macro_china_ppi_yearly",
    "macro_china_money_supply",
    "macro_china_shrzgm",
    "macro_china_new_financial_credit",
    "macro_china_lpr",
    "macro_china_consumer_goods_retail",
    "macro_china_industrial_production_yoy",
    "macro_china_gyzjz",
    "macro_china_gdzctz",
    "macro_china_exports_yoy",
    "macro_china_imports_yoy",
    "macro_china_trade_balance",
    "macro_china_hgjck",
    "macro_china_xfzxx",
    "macro_china_society_electricity",
    "macro_china_gdp",
    "macro_china_gdp_yearly",
    "macro_china_real_estate",
    "macro_china_enterprise_boom_index",
    "macro_china_fx_gold",
    "macro_china_urban_unemployment",
    "macro_china_czsr",
    "macro_china_qyspjg",
    "macro_china_stock_market_cap",
    "macro_china_commodity_price_index",
    "macro_china_society_traffic_volume",
    "macro_china_central_bank_balance",
    "macro_china_bank_financing",
    "macro_china_retail_price_index",
    "macro_china_market_margin_sh",
]


Q3_BOND_INTERFACE = "bond_zh_us_rate"


Q3_PROFIT_INTERFACE = "macro_china_nbs_nation"


Q3_PANEL_START = "2010-01"


Q3_REQUIRED = ["pmi", "gdp_yoy", "cgb10y", "spread_10y_2y", "shibor_3m"]


Q3_RAW_INSPECTION = {
    "macro_china_pmi": ("月份", "YYYY年MM月份", "monthly", "reference month", "yes"),
    "macro_china_non_man_pmi": ("日期", "YYYY-MM-DD", "monthly", "announcement date", "stale"),
    "macro_china_cx_pmi_yearly": ("日期", "YYYY-MM-DD", "monthly", "announcement date", "stale"),
    "macro_china_cx_services_pmi_yearly": ("日期", "YYYY-MM-DD", "monthly", "announcement date", "stale"),
    "macro_china_cpi": ("月份", "YYYY年MM月份", "monthly", "reference month", "yes"),
    "macro_china_cpi_yearly": ("日期", "YYYY-MM-DD", "monthly", "announcement date", "stale"),
    "macro_china_ppi": ("月份", "YYYY年MM月份", "monthly", "reference month", "yes"),
    "macro_china_ppi_yearly": ("日期", "YYYY-MM-DD", "monthly", "announcement date", "stale"),
    "macro_china_money_supply": ("月份", "YYYY年MM月份", "monthly", "reference month", "yes"),
    "macro_china_shrzgm": ("月份", "YYYYMM", "monthly", "reference month", "stale"),
    "macro_china_new_financial_credit": ("月份", "YYYY年MM月份", "monthly", "reference month", "yes"),
    "macro_china_lpr": ("TRADE_DATE", "YYYY-MM-DD", "monthly", "reference month", "yes"),
    "macro_china_consumer_goods_retail": ("月份", "YYYY年MM月份", "monthly", "reference month", "yes"),
    "macro_china_industrial_production_yoy": ("日期", "YYYY-MM-DD", "monthly", "announcement date", "stale"),
    "macro_china_gyzjz": ("月份", "YYYY年MM月份", "monthly", "reference month", "yes"),
    "macro_china_gdzctz": ("月份", "YYYY年MM月份", "monthly", "reference month", "yes"),
    "macro_china_exports_yoy": ("日期", "YYYY-MM-DD", "monthly", "announcement date", "stale"),
    "macro_china_imports_yoy": ("日期", "YYYY-MM-DD", "monthly", "announcement date", "stale"),
    "macro_china_trade_balance": ("日期", "YYYY-MM-DD", "monthly", "announcement date", "stale"),
    "macro_china_hgjck": ("月份", "YYYY年MM月份", "monthly", "reference month", "yes"),
    "macro_china_xfzxx": ("月份", "YYYY年MM月份", "monthly", "reference month", "yes"),
    "macro_china_society_electricity": ("统计时间", "YYYY.M", "monthly", "reference month", "yes"),
    "macro_china_gdp": ("季度", "YYYY年第Q季度", "quarterly", "reference quarter", "yes"),
    "macro_china_gdp_yearly": ("日期", "YYYY-MM-DD", "quarterly", "announcement date", "yes"),
    "macro_china_real_estate": ("日期", "YYYY-MM-01", "monthly", "reference month", "stale"),
    "macro_china_enterprise_boom_index": ("季度", "YYYY年第Q季度", "quarterly", "reference quarter", "unused"),
    "macro_china_fx_gold": ("月份", "YYYY年MM月份", "monthly", "reference month", "yes"),
    "macro_china_urban_unemployment": ("date", "YYYYMM", "monthly", "reference month", "yes"),
    "macro_china_czsr": ("月份", "YYYY年MM月份", "monthly", "reference month", "yes"),
    "macro_china_qyspjg": ("月份", "YYYY年MM月份", "monthly", "reference month", "yes"),
    "macro_china_stock_market_cap": ("数据日期", "YYYY年MM月份", "monthly", "reference month", "yes"),
    "macro_china_commodity_price_index": ("日期", "YYYY-MM-DD", "daily/weekly", "reference date", "yes"),
    "macro_china_society_traffic_volume": ("统计时间", "unknown", "unknown", "unknown", "failed"),
    "macro_china_central_bank_balance": ("统计时间", "YYYY.M", "monthly", "reference month", "yes"),
    "macro_china_bank_financing": ("日期", "YYYY-MM-01", "monthly", "reference month", "unused"),
    "macro_china_retail_price_index": ("统计月份", "YYYY.M", "monthly", "reference month", "stale"),
    "macro_china_market_margin_sh": ("日期", "YYYY-MM-DD", "daily", "reference date", "yes"),
    "bond_zh_us_rate": ("日期", "YYYY-MM-DD", "daily", "reference date", "yes"),
    "macro_china_nbs_industrial_profit": ("2026年9月", "YYYY年M月", "monthly", "reference month", "yes"),
}


def normalize_futures(frame: pd.DataFrame) -> pd.DataFrame:
    chinese = {
        "日期": "date",
        "开盘价": "open",
        "最高价": "high",
        "最低价": "low",
        "收盘价": "close",
        "成交量": "volume",
        "持仓量": "oi",
    }
    english = {
        "date": "date",
        "open": "open",
        "high": "high",
        "low": "low",
        "close": "close",
        "volume": "volume",
        "hold": "oi",
        "oi": "oi",
    }
    mapping = chinese if "日期" in frame.columns else english
    missing = [column for column in mapping if column not in frame.columns]
    if missing:
        raise ValueError(f"futures raw missing columns: {missing}")
    out = frame[list(mapping)].rename(columns=mapping).copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    for column in FUTURES_COLUMNS[1:]:
        out[column] = pd.to_numeric(out[column], errors="coerce")
    out = out.dropna(subset=["date", "close"]).drop_duplicates("date", keep="last")
    return out.sort_values("date").reset_index(drop=True)[FUTURES_COLUMNS]


def verify_futures(code: str, frame: pd.DataFrame) -> dict:
    duplicates = int(frame.duplicated("date").sum())
    price_columns = ["open", "high", "low", "close"]
    nonpositive = int((frame[price_columns] <= 0).sum().sum())
    missing_prices = int(frame[price_columns].isna().sum().sum())
    start = pd.Timestamp(frame["date"].min())
    end = pd.Timestamp(frame["date"].max())
    span_days = int((end - start).days)
    years = span_days / 365.25
    checks = {
        "dataset": code,
        "rows": int(len(frame)),
        "start": start.date().isoformat(),
        "end": end.date().isoformat(),
        "span_days": span_days,
        "years": years,
        "duplicate_dates": duplicates,
        "nonpositive_prices": nonpositive,
        "missing_prices": missing_prices,
        "at_least_5_years": bool(years >= 5.0),
        "no_duplicate_dates": bool(duplicates == 0),
        "positive_prices": bool(nonpositive == 0 and missing_prices == 0),
    }
    return checks


def fetch_futures() -> tuple[list[tuple[str, str, pd.DataFrame]], list[dict]]:
    import akshare as ak

    RAW.mkdir(parents=True, exist_ok=True)
    CLEAN.mkdir(parents=True, exist_ok=True)
    results: list[tuple[str, str, pd.DataFrame]] = []
    assertions: list[dict] = []
    for code, symbol in FUTURES_SYMBOLS.items():
        raw_path = RAW / f"futures_{code}.csv"
        source_path = RAW / f"futures_{code}_source.txt"
        source = "cached"
        if raw_path.exists() and raw_path.stat().st_size > 0:
            if source_path.exists():
                source = source_path.read_text(encoding="utf-8").strip()
            record(f"Futures {code}", "skipped", f"using cached source={source}", str(raw_path))
            log.info("Futures %s skipped; cached source=%s", code, source)
            raw = pd.read_csv(raw_path)
        else:
            attempts = [
                (
                    "akshare.futures_main_sina",
                    lambda: ak.futures_main_sina(
                        symbol=symbol, start_date="20050101", end_date="22220101"
                    ),
                ),
                ("akshare.futures_zh_daily_sina", lambda: ak.futures_zh_daily_sina(symbol=symbol)),
            ]
            failures = []
            raw = None
            for name, call in attempts:
                try:
                    with time_limit(180):
                        frame = call()
                    if frame is None or len(frame) == 0:
                        raise ValueError("empty response")
                    frame.to_csv(raw_path, index=False, encoding="utf-8")
                    source_path.write_text(name, encoding="utf-8")
                    source = name
                    raw = frame
                    record(f"Futures {code}", "ok", f"source={name} rows={len(frame)}", str(raw_path))
                    log.info("Futures %s source used: %s (%d rows)", code, name, len(frame))
                    break
                except Exception as exc:
                    message = f"{type(exc).__name__}: {exc}"
                    failures.append(f"{name}: {message}")
                    log.error("Futures %s %s failed: %s", code, name, message)
            if raw is None:
                record(f"Futures {code}", "failed", " | ".join(failures), str(raw_path))
                assertions.append(
                    {
                        "dataset": code,
                        "rows": 0,
                        "start": None,
                        "end": None,
                        "span_days": 0,
                        "years": 0.0,
                        "duplicate_dates": 0,
                        "nonpositive_prices": 0,
                        "missing_prices": 0,
                        "at_least_5_years": False,
                        "no_duplicate_dates": False,
                        "positive_prices": False,
                    }
                )
                continue
        clean = normalize_futures(raw)
        clean_path = CLEAN / f"futures_{code}.csv"
        clean.to_csv(clean_path, index=False, encoding="utf-8")
        checks = verify_futures(code, clean)
        assertions.append(checks)
        results.append((code, source, clean))
        log.info(
            "Futures %s clean rows=%d %s..%s years=%.2f",
            code,
            checks["rows"],
            checks["start"],
            checks["end"],
            checks["years"],
        )
    return results, assertions


def update_futures_metadata(results: list[tuple[str, str, pd.DataFrame]], assertions: list[dict]):
    if not METADATA.exists():
        raise FileNotFoundError(f"metadata missing: {METADATA}")
    metadata = json.loads(METADATA.read_text(encoding="utf-8"))
    artifacts = metadata.setdefault("artifacts", {})
    for code, source, frame in results:
        relative = f"data/clean/futures_{code}.csv"
        artifacts[relative] = build_artifact(
            ROOT / relative,
            f"Sina main continuous futures {FUTURES_SYMBOLS[code]} ({code}); interface={source}",
            source,
            FUTURES_COLUMNS,
            len(frame),
        )
    metadata["futures_assertions"] = assertions
    metadata["updated_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
    METADATA.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")


def print_futures_report(results: list[tuple[str, str, pd.DataFrame]], assertions: list[dict]):
    table = Table(title="Q2 futures snapshots")
    table.add_column("Dataset")
    table.add_column("Source")
    table.add_column("Rows", justify="right")
    table.add_column("Start")
    table.add_column("End")
    table.add_column("Years", justify="right")
    for code, source, frame in results:
        table.add_row(
            code,
            source,
            f"{len(frame):,}",
            str(frame["date"].iloc[0]),
            str(frame["date"].iloc[-1]),
            f"{(pd.Timestamp(frame['date'].iloc[-1]) - pd.Timestamp(frame['date'].iloc[0])).days / 365.25:.2f}",
        )
    console.print(table)

    check_table = Table(title="Futures assertions")
    check_table.add_column("Dataset")
    check_table.add_column("5y")
    check_table.add_column("No duplicate dates")
    check_table.add_column("Positive prices")
    for item in assertions:
        check_table.add_row(
            item["dataset"],
            "PASS" if item["at_least_5_years"] else "FAIL",
            "PASS" if item["no_duplicate_dates"] else "FAIL",
            "PASS" if item["positive_prices"] else "FAIL",
        )
    console.print(check_table)


def futures_only():
    RAW.mkdir(parents=True, exist_ok=True)
    CLEAN.mkdir(parents=True, exist_ok=True)
    futures_results, futures_assertions = fetch_futures()
    update_futures_metadata(futures_results, futures_assertions)
    print_futures_report(futures_results, futures_assertions)
    failed = [
        f"{item['dataset']} futures"
        for item in futures_assertions
        if not (
            item["at_least_5_years"]
            and item["no_duplicate_dates"]
            and item["positive_prices"]
        )
    ]
    if failed:
        log.error("assertion failures: %s", failed)
        return 1
    log.info("futures fetch complete; all assertions passed")
    return 0


def offline():
    if not METADATA.exists():
        log.error("metadata missing: %s", METADATA)
        return 1
    metadata = json.loads(METADATA.read_text(encoding="utf-8"))
    artifacts = metadata.get("artifacts", {})
    if not artifacts:
        log.error("metadata contains no artifacts")
        return 1
    failed = []
    for relative, info in artifacts.items():
        path = ROOT / relative
        if not path.exists():
            log.error("OFFLINE missing %s", relative)
            failed.append(relative)
            continue
        actual = sha256(path)
        expected = info.get("sha256")
        if actual != expected:
            log.error("OFFLINE checksum mismatch %s expected=%s actual=%s", relative, expected, actual)
            failed.append(relative)
        else:
            log.info("OFFLINE OK %s rows=%s sha256=%s", relative, info.get("rows"), actual)
    if failed:
        return 1
    log.info("offline verification complete; all checksums match")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Fetch Q2 futures data")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--offline", action="store_true", help="verify clean artifacts against metadata checksums")
    group.add_argument("--futures", action="store_true", help="fetch or refresh only the Q2 futures snapshots")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.offline:
        return offline()
    return futures_only()


if __name__ == "__main__":
    raise SystemExit(main())
