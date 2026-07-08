"""
Time series anomaly detection over daily transaction KPIs.

Reads main_marts.mart_daily_kpis straight out of DuckDB (built by dbt) and
flags days where volume, dollar amount, or decline rate deviate sharply from
a seasonally-aware historical baseline.

Method: seasonal rolling z-score.
    Because the data has real weekly seasonality (weekend lift baked into
    the generator), a naive trailing-window mean/std would misfire every
    weekend. Instead, for each day we compare it only to the same day-of-week
    over the preceding `LOOKBACK_WEEKS` occurrences ("the last 6 Mondays"),
    which cancels out the weekly pattern and isolates genuine regime shifts.
    This is a lighter-weight alternative to STL decomposition that's easy to
    reason about and works well with ~18 months of daily data.

The detector has no knowledge of the anomalies injected by
python/etl/generate_synthetic_data.py -- it only sees the aggregated KPI
series and flags statistical outliers. Plain-English descriptions are then
assembled from heuristics over the flagged pattern (single-day system-wide
drop vs. multi-day regional spike with high declines vs. sustained broad
increase), not from the injected event labels themselves.

Run:
    python python/anomaly_detection/detect_anomalies.py
"""

from __future__ import annotations

import duckdb
import numpy as np
import pandas as pd

DB_PATH = "warehouse/transaction_analytics.duckdb"
OUTPUT_PATH = "outputs/anomalies.csv"

LOOKBACK_WEEKS = 10         # how many same-weekday occurrences to baseline against
MIN_PERIODS = 6             # minimum prior occurrences required before flagging
Z_THRESHOLD = 4.0           # |z| below this is not flagged at all
MIN_PCT_DEVIATION = 0.25    # also require a >=25% move so tiny-baseline noise doesn't flag


# --------------------------------------------------------------------------
# Core detection: seasonal rolling z-score
# --------------------------------------------------------------------------

def seasonal_rolling_zscore(df: pd.DataFrame, date_col: str, value_col: str) -> pd.DataFrame:
    """
    For each row, compute a z-score against the trailing same-weekday
    baseline (mean/std of the same weekday over the prior LOOKBACK_WEEKS
    occurrences, excluding the current day itself to avoid leakage).
    """
    out = df[[date_col, value_col]].copy().sort_values(date_col).reset_index(drop=True)
    out["dow"] = pd.to_datetime(out[date_col]).dt.dayofweek

    out["baseline_mean"] = np.nan
    out["baseline_std"] = np.nan

    for dow in range(7):
        mask = out["dow"] == dow
        vals = out.loc[mask, value_col]
        # shift(1) excludes the current day from its own baseline
        roll_mean = vals.shift(1).rolling(window=LOOKBACK_WEEKS, min_periods=MIN_PERIODS).mean()
        roll_std = vals.shift(1).rolling(window=LOOKBACK_WEEKS, min_periods=MIN_PERIODS).std()
        out.loc[mask, "baseline_mean"] = roll_mean
        out.loc[mask, "baseline_std"] = roll_std

    # Guard against a near-zero std collapsing the z-score to +/-inf.
    floor = out["baseline_mean"].abs() * 0.05
    safe_std = out["baseline_std"].where(out["baseline_std"] > floor, floor)
    safe_std = safe_std.replace(0, np.nan)

    out["zscore"] = (out[value_col] - out["baseline_mean"]) / safe_std
    out["pct_deviation"] = (out[value_col] - out["baseline_mean"]) / out["baseline_mean"]
    return out


def severity_for(z: float) -> str | None:
    az = abs(z)
    if az >= 8.0:
        return "critical"
    if az >= 5.5:
        return "high"
    if az >= Z_THRESHOLD:
        return "medium"
    return None


def flag_anomalies(scored: pd.DataFrame, date_col: str, value_col: str) -> pd.DataFrame:
    flagged = scored[
        scored["zscore"].notna()
        & (scored["zscore"].abs() >= Z_THRESHOLD)
        & (scored["pct_deviation"].abs() >= MIN_PCT_DEVIATION)
    ].copy()
    flagged["severity"] = flagged["zscore"].apply(severity_for)
    flagged = flagged[flagged["severity"].notna()]
    return flagged[[date_col, value_col, "baseline_mean", "zscore", "pct_deviation", "severity"]]


# --------------------------------------------------------------------------
# Plain-English description heuristics
# --------------------------------------------------------------------------

def describe(row: pd.Series, metric: str, region: str | None, dim_label: str, decline_flags: set) -> str:
    start = pd.to_datetime(row["start_date"]).date()
    end = pd.to_datetime(row["end_date"]).date()
    when = f"{start}" if start == end else f"{start} to {end} ({row['n_days']} days)"
    peak_date = pd.to_datetime(row["transaction_date"]).date()

    z = row["zscore"]
    pct = row["pct_deviation"] * 100
    direction = "above" if z > 0 else "below"
    scope = f"{region} {dim_label}" if region else "system-wide"

    if metric == "txn_count" and z < 0 and abs(pct) > 60:
        return (
            f"{when}: {scope} transaction volume collapsed to {row['txn_count']:.0f} on its worst day "
            f"({peak_date}), {abs(pct):.0f}% below the {row['baseline_mean']:.0f}-transaction baseline "
            f"(peak z={z:.1f}) -- pattern consistent with a processing outage or service disruption."
        )

    if metric == "txn_count" and z > 0 and region and dim_label == "region" and (peak_date, region) in decline_flags:
        return (
            f"{when}: {scope} transaction volume spiked to {row['txn_count']:.0f} at its peak "
            f"({peak_date}, {pct:.0f}% {direction} baseline, z={z:.1f}) alongside an elevated decline "
            f"rate -- pattern consistent with a fraud attempt or attack concentrated in this region."
        )

    if metric == "total_amount" and z > 0 and not region:
        return (
            f"{when}: system-wide transaction amount peaked at ${row['total_amount']:,.0f} on {peak_date} "
            f"({pct:.0f}% {direction} the ${row['baseline_mean']:,.0f} baseline, z={z:.1f}) "
            f"-- consistent with a broad demand surge (e.g. holiday spending)."
        )

    return (
        f"{when}: {scope} {metric.replace('_', ' ')} peaked at {pct:+.0f}% vs. its historical "
        f"same-weekday baseline on {peak_date} (z={z:.1f})."
    )


# --------------------------------------------------------------------------
# Event consolidation: collapse consecutive flagged days per metric/region
# into a single event row, anchored on the day with the largest |z|.
# --------------------------------------------------------------------------

def consolidate_events(df: pd.DataFrame, date_col: str = "transaction_date") -> pd.DataFrame:
    df = df.copy()
    df["region_key"] = df["region"].fillna("__none__")
    events = []

    for _, grp in df.groupby(["metric", "region_key"]):
        grp = grp.sort_values(date_col).reset_index(drop=True)
        dates = pd.to_datetime(grp[date_col])
        new_run = (dates.diff().dt.days != 1).cumsum()
        for _, run in grp.groupby(new_run):
            peak = run.loc[run["zscore"].abs().idxmax()].copy()
            peak["start_date"] = run[date_col].min()
            peak["end_date"] = run[date_col].max()
            peak["n_days"] = len(run)
            events.append(peak)

    out = pd.DataFrame(events).drop(columns=["region_key"])
    return out.sort_values("start_date").reset_index(drop=True)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main() -> None:
    con = duckdb.connect(DB_PATH, read_only=True)
    kpis = con.execute("SELECT * FROM main_marts.mart_daily_kpis").df()
    con.close()
    kpis["transaction_date"] = pd.to_datetime(kpis["transaction_date"])

    all_flagged = []

    # ---- system-wide series ----
    system_daily = (
        kpis.groupby("transaction_date", as_index=False)
        .agg(txn_count=("txn_count", "sum"), total_amount=("total_amount", "sum"))
    )
    for metric in ["txn_count", "total_amount"]:
        scored = seasonal_rolling_zscore(system_daily, "transaction_date", metric)
        flagged = flag_anomalies(scored, "transaction_date", metric)
        flagged["metric"] = f"system_{metric}"
        flagged["region"] = None
        all_flagged.append(flagged)

    # ---- region-level series (volume + decline rate, to catch fraud concentrated in one region) ----
    region_daily = (
        kpis.groupby(["transaction_date", "region"], as_index=False)
        .agg(
            txn_count=("txn_count", "sum"),
            total_amount=("total_amount", "sum"),
            declined_count=("declined_count", "sum"),
        )
    )
    region_daily["declined_rate"] = region_daily["declined_count"] / region_daily["txn_count"]

    # Decline rate is noisy at daily/region grain (small sample sizes), so it
    # isn't run through the formal z-score flagger as its own metric. Instead
    # it's used as supporting context: any (date, region) where the decline
    # rate is at least 50% above that region's trailing 10-week average is
    # treated as "elevated declines" and folded into the description of a
    # co-occurring volume anomaly (the fraud-spike signature is a volume
    # spike *and* a decline-rate spike happening together).
    decline_flags: set[tuple] = set()
    region_volume_frames = []
    for region, grp in region_daily.groupby("region"):
        grp = grp.sort_values("transaction_date")
        baseline_decline = grp["declined_rate"].shift(1).rolling(window=70, min_periods=14).mean()
        elevated = grp["declined_rate"] > (1.5 * baseline_decline)
        for d in grp.loc[elevated.fillna(False), "transaction_date"]:
            decline_flags.add((pd.Timestamp(d).date(), region))

        scored = seasonal_rolling_zscore(grp, "transaction_date", "txn_count")
        flagged = flag_anomalies(scored, "transaction_date", "txn_count")
        flagged["metric"] = "region_txn_count"
        flagged["region"] = region
        region_volume_frames.append(flagged)

    all_flagged.extend(region_volume_frames)

    # ---- channel-level series (volume only) ----
    channel_daily = (
        kpis.groupby(["transaction_date", "channel"], as_index=False)
        .agg(txn_count=("txn_count", "sum"), total_amount=("total_amount", "sum"))
    )
    channel_frames = []
    for channel, grp in channel_daily.groupby("channel"):
        scored = seasonal_rolling_zscore(grp, "transaction_date", "txn_count")
        flagged = flag_anomalies(scored, "transaction_date", "txn_count")
        flagged["metric"] = "channel_txn_count"
        flagged["region"] = channel
        channel_frames.append(flagged)
    all_flagged.extend(channel_frames)

    combined = pd.concat(all_flagged, ignore_index=True, sort=False)
    events = consolidate_events(combined)

    # ---- build descriptions from the peak day of each consolidated event ----
    descriptions = []
    for _, row in events.iterrows():
        base_metric = row["metric"].replace("system_", "").replace("region_", "").replace("channel_", "")
        dim_label = "channel" if row["metric"].startswith("channel_") else "region"
        region = row["region"] if row["metric"].startswith(("region_", "channel_")) else None
        descriptions.append(describe(row, base_metric, region, dim_label, decline_flags))
    events["description"] = descriptions

    result = events.rename(columns={"start_date": "date"})[
        ["date", "end_date", "n_days", "metric", "region", "zscore", "pct_deviation", "severity", "description"]
    ]
    result["date"] = result["date"].dt.date
    result["end_date"] = result["end_date"].dt.date
    result["zscore"] = result["zscore"].round(2)
    result["pct_deviation"] = (result["pct_deviation"] * 100).round(1)
    result = result.sort_values("date").reset_index(drop=True)

    result.to_csv(OUTPUT_PATH, index=False)
    print(f"Flagged {len(result)} anomaly events -> {OUTPUT_PATH}")
    print(result["severity"].value_counts())
    print("\nEvents by metric:")
    print(result.groupby("metric")["date"].agg(["min", "max", "count"]))


if __name__ == "__main__":
    main()
