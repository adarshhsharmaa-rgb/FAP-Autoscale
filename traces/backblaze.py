"""
traces/backblaze.py  --  Backblaze Drive Stats (Q2 2026): real hardware failures.

Source  : https://www.backblaze.com/cloud-storage/resources/hard-drive-test-data
Archive : data_Q2_2026.zip (2.09 GB, 91 daily CSVs, one row per drive per day,
          197 columns: drive identity, a `failure` flag and raw/normalised
          S.M.A.R.T. attributes).
Terms   : free to use; cite Backblaze as the source; the data may not be sold.

FAP-Scale uses it as ground truth for node failure: every simulated cluster
node replays the health telemetry of one real drive, and the node goes down on
the day that drive was recorded as failed.

Building the sample
-------------------
`build_sample()` reads the zip over HTTP Range requests (see http_range.py),
walking the days newest-first. A drive's last row carries `failure = 1`, so by
the time an earlier day is read we already know which drives go on to fail. We
keep

    * every drive that fails inside the window (all of its days), and
    * a deterministic ~1.2 % hash sample of the drives that do not.

The result (~270 k drive-days) is written to data/processed/ and committed, so
nobody else has to pull 1.4 GB. Because healthy drives are sub-sampled, the
failure rate in the sample is far above the fleet rate; `HEALTHY_SAMPLE_RATE`
is stored so population-level metrics can be recovered with inverse weights.

    python -m traces.backblaze --days 61          # rebuild the committed sample

The build checkpoints after every day (in the system temp directory), so if the
connection drops it resumes where it stopped when run again.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import tempfile
import time
import zipfile
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from traces.http_range import HTTPRangeFile

QUARTER = "Q2_2026"
ARCHIVE_URL = f"https://f001.backblazeb2.com/file/Backblaze-Hard-Drive-Data/data_{QUARTER}.zip"
SOURCE_PAGE = "https://www.backblaze.com/cloud-storage/resources/hard-drive-test-data"

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(_ROOT, "data", "processed")
SAMPLE_PATH = os.path.join(PROCESSED_DIR, f"backblaze_{QUARTER.lower()}_sample.csv.gz")
META_PATH = os.path.join(PROCESSED_DIR, f"backblaze_{QUARTER.lower()}_sample.meta.json")

# S.M.A.R.T. attributes kept. 5/187/188/197/198 are the five Backblaze itself
# reports as predictive of failure; 194 is temperature, 199 interface CRC
# errors, 9 power-on hours.
SMART = {
    "smart_5_raw": "realloc_sectors",
    "smart_9_raw": "power_on_hours",
    "smart_187_raw": "reported_uncorrectable",
    "smart_188_raw": "command_timeouts",
    "smart_194_raw": "temperature_c",
    "smart_197_raw": "pending_sectors",
    "smart_198_raw": "offline_uncorrectable",
    "smart_199_raw": "crc_errors",
}
ID_COLS = ["date", "serial_number", "model", "failure", "datacenter", "vault_id", "pod_id"]
HEALTHY_SAMPLE_PER_MILLE = 12   # 1.2 % of non-failing drives
HEALTHY_SAMPLE_RATE = HEALTHY_SAMPLE_PER_MILLE / 1000.0

_DAY_RE = re.compile(r"(\d{4}-\d{2}-\d{2})\.csv$")


def _in_healthy_sample(serials: pd.Series) -> np.ndarray:
    """Stable hash sample: the same serial is always in or always out."""
    h = pd.util.hash_array(serials.to_numpy(dtype=object), categorize=False)
    return (h % np.uint64(1000)) < np.uint64(HEALTHY_SAMPLE_PER_MILLE)


def build_sample(days: int = 61, out_path: str = SAMPLE_PATH, verbose: bool = True,
                 work_dir: Optional[str] = None) -> pd.DataFrame:
    """Stream the newest `days` daily files and write the reduced sample."""
    t0 = time.time()
    remote = HTTPRangeFile(ARCHIVE_URL)
    zf = zipfile.ZipFile(remote)
    members = sorted(
        (m for m in zf.infolist() if _DAY_RE.search(m.filename) and "__MACOSX" not in m.filename),
        key=lambda m: _DAY_RE.search(m.filename).group(1), reverse=True)[:days]
    if verbose:
        mb = sum(m.compress_size for m in members) / 1e6
        print(f"{QUARTER}: {len(members)} daily files, {mb:,.0f} MB to stream "
              f"({members[-1].filename[-14:-4]} .. {members[0].filename[-14:-4]})")

    usecols = ID_COLS + list(SMART)
    str_cols = {c: str for c in ("serial_number", "model", "datacenter", "vault_id", "pod_id")}
    work = work_dir or os.path.join(tempfile.gettempdir(), f"fap-scale-backblaze-{QUARTER}-{days}d")
    os.makedirs(work, exist_ok=True)
    state_path = os.path.join(work, "state.json")
    state = {"failed": [], "fleet_sizes": {}}
    if os.path.exists(state_path):
        with open(state_path) as f:
            state = json.load(f)
    failed: set = set(state["failed"])
    fleet_sizes: Dict[str, int] = dict(state["fleet_sizes"])
    parts: List[pd.DataFrame] = []
    for i, m in enumerate(members):
        day = _DAY_RE.search(m.filename).group(1)
        part_path = os.path.join(work, f"{day}.pkl")
        if day in fleet_sizes and os.path.exists(part_path):
            parts.append(pd.read_pickle(part_path))     # finished in an earlier run
            continue
        for attempt in range(4):
            try:
                with zf.open(m) as fh:
                    df = pd.read_csv(fh, usecols=usecols, dtype=str_cols)
                break
            except (IOError, zipfile.BadZipFile, pd.errors.ParserError) as exc:
                if attempt == 3:
                    raise
                print(f"  {day}: {type(exc).__name__}, retrying", flush=True)
                time.sleep(5 * (attempt + 1))
        failed.update(df.loc[df["failure"] == 1, "serial_number"])
        keep = df["serial_number"].isin(failed).to_numpy() | _in_healthy_sample(df["serial_number"])
        part = df.loc[keep]
        parts.append(part)
        fleet_sizes[day] = len(df)
        # checkpoint: the kept rows first, then the state that marks the day done
        part.to_pickle(part_path)
        with open(state_path + ".tmp", "w") as f:
            json.dump({"failed": sorted(failed), "fleet_sizes": fleet_sizes}, f)
        os.replace(state_path + ".tmp", state_path)
        if verbose:
            done = remote.bytes_fetched / 1e6
            print(f"  [{i + 1:2d}/{len(members)}] {day}  fleet={len(df):,}  kept={int(keep.sum()):,}  "
                  f"failed so far={len(failed):,}  {done:,.0f} MB  {time.time() - t0:,.0f}s", flush=True)

    out = pd.concat(parts, ignore_index=True).rename(columns=SMART)
    out["will_fail"] = out["serial_number"].isin(failed)
    # anonymous integer ids instead of serial numbers
    order = {s: k for k, s in enumerate(sorted(out["serial_number"].unique()))}
    out.insert(0, "drive_id", out["serial_number"].map(order).astype(np.int32))
    out = out.drop(columns="serial_number").sort_values(["drive_id", "date"]).reset_index(drop=True)
    for c in SMART.values():
        out[c] = out[c].astype("Int64")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    out.to_csv(out_path, index=False, compression={"method": "gzip", "compresslevel": 9, "mtime": 0})
    meta = {
        "source": "Backblaze Drive Stats", "quarter": QUARTER, "archive_url": ARCHIVE_URL,
        "source_page": SOURCE_PAGE,
        "first_day": members[-1].filename[-14:-4], "last_day": members[0].filename[-14:-4],
        "days": len(members),
        "fleet_size_last_day": fleet_sizes[members[0].filename[-14:-4]],
        "drive_days_scanned": int(sum(fleet_sizes.values())),
        "failed_drives": len(failed),
        "healthy_drives_sampled": int(out.loc[~out["will_fail"], "drive_id"].nunique()),
        "healthy_sample_rate": HEALTHY_SAMPLE_RATE,
        "rows": len(out),
        "columns": list(out.columns),
    }
    with open(META_PATH, "w") as f:
        json.dump(meta, f, indent=2)
    if verbose:
        print(f"wrote {out_path} ({os.path.getsize(out_path) / 1e6:.1f} MB, {len(out):,} rows) "
              f"in {time.time() - t0:,.0f}s")
    return out


def load_meta() -> dict:
    with open(META_PATH) as f:
        return json.load(f)


def load_drive_days(path: Optional[str] = None) -> pd.DataFrame:
    """
    Load the committed sample: one row per drive per day.

    Adds `days_to_failure` (0 on the failure day, NaN for drives that survive
    the window) so labels of the form "fails within H days" are one comparison.
    """
    path = path or SAMPLE_PATH
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} is missing. Rebuild it with:  python -m traces.backblaze")
    df = pd.read_csv(path, parse_dates=["date"], dtype={"model": str, "datacenter": str,
                                                         "vault_id": str, "pod_id": str})
    fail_day = df.loc[df["failure"] == 1].groupby("drive_id")["date"].max()
    df["days_to_failure"] = (df["drive_id"].map(fail_day) - df["date"]).dt.days
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Rebuild the Backblaze Drive Stats sample.")
    ap.add_argument("--days", type=int, default=61, help="newest N days of the quarter")
    args = ap.parse_args()
    build_sample(days=args.days)
