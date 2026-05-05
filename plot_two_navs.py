from pathlib import Path
import re
import sys

import pandas as pd
import matplotlib.pyplot as plt


EXPERIMENTS = [
    {
        "folder": "sparse_switch_mv_nifty500_20260505_023508",
        "label": "Sparse switching mean-variance",
    },
    {
        "folder": "sparse_switch_mv_nifty500_20260505_023519",
        "label": "Baseline",
    },
]
TICKS_PER_DAY = 375


def find_experiment_dir(root: Path, exp_name: str) -> Path:
    matches = [p for p in root.rglob(exp_name) if p.is_dir()]

    if not matches:
        raise FileNotFoundError(f"Could not find experiment folder: {exp_name}")

    if len(matches) > 1:
        print(f"[WARN] Multiple matches for {exp_name}. Using: {matches[0]}")

    return matches[0]


def find_column_case_insensitive(columns, candidates):
    lower_map = {str(c).lower(): c for c in columns}

    for candidate in candidates:
        if candidate in columns:
            return candidate

        candidate_lower = candidate.lower()
        if candidate_lower in lower_map:
            return lower_map[candidate_lower]

    return None


def convert_to_daily(x, nav):
    """
    If x is datetime-like, take last NAV per calendar day.
    If x is tick index, take last NAV every TICKS_PER_DAY rows.
    """

    df = pd.DataFrame({
        "x": x.reset_index(drop=True),
        "nav": nav.reset_index(drop=True),
    })

    df = df.dropna(subset=["nav"])

    if pd.api.types.is_datetime64_any_dtype(df["x"]):
        df["day"] = df["x"].dt.date
        daily = df.groupby("day", as_index=False)["nav"].last()
        return pd.to_datetime(daily["day"]), daily["nav"]

    df["day"] = df.index // TICKS_PER_DAY
    daily = df.groupby("day", as_index=False)["nav"].last()

    return daily["day"], daily["nav"]


def load_nav_from_csv_or_parquet(exp_dir: Path):
    files = list(exp_dir.rglob("*.csv")) + list(exp_dir.rglob("*.parquet"))

    preferred_keywords = [
        "nav",
        "equity",
        "portfolio",
        "metrics",
        "result",
        "curve",
    ]

    files = sorted(
        files,
        key=lambda p: (
            not any(k in p.name.lower() for k in preferred_keywords),
            len(str(p)),
        ),
    )

    nav_col_candidates = [
        "nav",
        "NAV",
        "portfolio_value",
        "equity",
        "equity_curve",
        "total_value",
        "value",
    ]

    time_col_candidates = [
        "timestamp",
        "time",
        "datetime",
        "date",
    ]

    for file in files:
        try:
            if file.suffix.lower() == ".csv":
                df = pd.read_csv(file)
            else:
                df = pd.read_parquet(file)
        except Exception:
            continue

        if df.empty:
            continue

        nav_col = find_column_case_insensitive(df.columns, nav_col_candidates)

        if nav_col is None:
            continue

        time_col = find_column_case_insensitive(df.columns, time_col_candidates)

        out = df.copy()

        if time_col is not None:
            out[time_col] = pd.to_datetime(out[time_col], errors="coerce")
            out = out.dropna(subset=[time_col])
            x = out[time_col].reset_index(drop=True)
        else:
            x = pd.Series(range(len(out)))

        y = pd.to_numeric(out[nav_col], errors="coerce").reset_index(drop=True)

        valid = y.notna()

        if valid.sum() > 0:
            print(f"[OK] Loaded NAV from: {file}")
            return x[valid].reset_index(drop=True), y[valid].reset_index(drop=True), file

    return None


def load_nav_from_logs(exp_dir: Path):
    log_files = list(exp_dir.rglob("*.log")) + list(exp_dir.rglob("*.txt"))

    rows = []

    pattern = re.compile(
        r"(?P<timestamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}).*?NAV=(?P<nav>[0-9.]+)"
    )

    for file in log_files:
        try:
            text = file.read_text(errors="ignore")
        except Exception:
            continue

        for m in pattern.finditer(text):
            rows.append(
                {
                    "timestamp": m.group("timestamp"),
                    "nav": float(m.group("nav")),
                    "source": file,
                }
            )

    if not rows:
        return None

    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp", "nav"])

    if df.empty:
        return None

    df = df.sort_values("timestamp")

    source = df["source"].iloc[0]
    print(f"[OK] Loaded NAV from logs under: {exp_dir}")

    return (
        df["timestamp"].reset_index(drop=True),
        df["nav"].reset_index(drop=True),
        source,
    )


def load_nav(exp_dir: Path):
    result = load_nav_from_csv_or_parquet(exp_dir)

    if result is not None:
        return result

    result = load_nav_from_logs(exp_dir)

    if result is not None:
        return result

    raise FileNotFoundError(
        f"No NAV data found in {exp_dir}. "
        "Expected a CSV/parquet with a NAV-like column or logs containing NAV=..."
    )


def plot_one_experiment(root: Path, exp):
    exp_name = exp["folder"]
    label = exp["label"]

    exp_dir = find_experiment_dir(root, exp_name)
    x, nav, source_file = load_nav(exp_dir)

    daily_x, daily_nav = convert_to_daily(x, nav)

    plt.figure(figsize=(13, 6))
    plt.plot(daily_x, daily_nav)

    plt.title(f"Daily NAV Curve: {label}")
    plt.xlabel("Trading Day")
    plt.ylabel("NAV")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    print(f"[PLOT] {label}")
    print(f"[FOLDER] {exp_name}")
    print(f"[SOURCE] {source_file}")
    print(f"[POINTS] Raw: {len(nav)}, Daily: {len(daily_nav)}")
    print("Close this plot window to open the next one.")

    plt.show()


def main():
    if len(sys.argv) > 1:
        root = Path(sys.argv[1]).resolve()
    else:
        root = Path.cwd().resolve()

    print(f"[ROOT] Searching from: {root}")

    for exp in EXPERIMENTS:
        plot_one_experiment(root, exp)


if __name__ == "__main__":
    main()