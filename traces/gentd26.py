"""
traces/gentd26.py  --  Alibaba GenTD26 (cluster-trace-v2026-GenAI).

Source : https://github.com/alibaba/clusterdata/tree/master/cluster-trace-v2026-GenAI
         (Alibaba Open Cluster Trace Program; SoCC '25 and EuroSys '26 papers)
What   : one production day of a Stable Diffusion serving system, sampled every
         57 s (1441 steps), across three layers:
           gateway   qps.csv, queue_size_raw_anon.csv, queue_rt_raw_anon.csv
           service   pipeline_inference_data_anon.csv   (end-to-end latency)
           infra     pod_gpu_duty_cycle_anon.csv, pod_memory_util_anon.csv
         plus lora_request_trace.csv, a 23-day request log.
         Values are anonymised by the publisher (time shift, metric scaling,
         hashed ids) with trends and correlations preserved.

The repository carries no licence beyond "use the traces for study or research
purposes", so nothing from it is committed here. `fetch()` downloads the files
from a pinned upstream commit into data/raw/gentd26/ (git-ignored) and checks
their SHA-256.

How FAP-Scale uses it
    workload      gateway QPS summed over containers  -> the load L(t) to scale for
    node load     each pod's GPU duty cycle / memory  -> utilisation of the host it runs on
    queueing      per-container gateway queue time    -> queue-pressure term of the CPI
    interference  per-container latency inflation     -> observed slowdown on that host
"""
from __future__ import annotations

import hashlib
import io
import os
import tarfile
import urllib.request
import warnings
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, List

import numpy as np
import pandas as pd

from traces.http_range import USER_AGENT

COMMIT = "8c132fb7717fbd44cf434cf81f2cd865a80072d5"   # alibaba/clusterdata, 2026-04-27
BASE_URL = f"https://raw.githubusercontent.com/alibaba/clusterdata/{COMMIT}/cluster-trace-v2026-GenAI"
SOURCE_PAGE = "https://github.com/alibaba/clusterdata/tree/master/cluster-trace-v2026-GenAI"

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(_ROOT, "data", "raw", "gentd26")

STEP_SECONDS = 57          # sampling period of every monitoring series
N_STEPS = 1441             # 22.8 hours

# local csv name -> (remote file, sha256 of the remote file)
FILES: Dict[str, tuple] = {
    "qps.csv": ("qps.tar.gz",
                "6bd1a21e3f916740df1e768fc43346148f1598b15fcc68c658a8faf1f87461c6"),
    "queue_size_raw_anon.csv": ("queue_size_raw_anon.tar.gz",
                                "585499916ed94900921e72c01213a618cad52443fc3f20f1e0e7ad4c4bb11798"),
    "queue_rt_raw_anon.csv": ("queue_rt_raw_anon.tar.gz",
                              "419deaccd1cc33ef3050b676f75a49af726c169de62ae29344ea2fdf1d4f3819"),
    "pod_gpu_duty_cycle_anon.csv": ("pod_gpu_duty_cycle_anon.tar.gz",
                                    "c4070218189b2811995c30dd9fefb8a793a83a54dccebe7b41cab2d123916855"),
    "pod_memory_util_anon.csv": ("pod_memory_util_anon.tar.gz",
                                 "0ccfb52821432dd6eed546ae772299c9c9fa9bbed952eb44953c41564afcfa7c"),
    "pipeline_inference_data_anon.csv": ("pipeline_inference_data_anon.tar.gz",
                                         "4899bd06c7089764046ae8b1ba5dfd243e47062c96d9ff34f7d344558602bbd5"),
    "lora_request_trace.csv": ("lora_request_trace.csv",
                               "263fb88f1095f62d360cf89ded8cfc24fa2c5d111c1b16e7aaa9ccbfc4b37d78"),
}


# --------------------------------------------------------------------------- #
# Download
# --------------------------------------------------------------------------- #
def _download(remote: str, sha256: str) -> bytes:
    req = urllib.request.Request(f"{BASE_URL}/{remote}", headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=120) as resp:
        blob = resp.read()
    got = hashlib.sha256(blob).hexdigest()
    if got != sha256:
        raise IOError(f"{remote}: checksum mismatch (got {got[:12]}..., expected {sha256[:12]}...)")
    return blob


def _csv_from_tar(blob: bytes, wanted: str) -> bytes:
    """Pull exactly one named regular file out of a .tar.gz, nothing else."""
    with tarfile.open(fileobj=io.BytesIO(blob), mode="r:gz") as tar:
        for member in tar:
            if member.isfile() and os.path.basename(member.name) == wanted:
                return tar.extractfile(member).read()
    raise IOError(f"{wanted} not found in archive")


def is_available() -> bool:
    return all(os.path.exists(os.path.join(RAW_DIR, name)) for name in FILES)


def fetch(force: bool = False, verbose: bool = True) -> str:
    """Download GenTD26 (about 5 MB over the wire) into data/raw/gentd26/."""
    os.makedirs(RAW_DIR, exist_ok=True)
    for local, (remote, sha256) in FILES.items():
        dest = os.path.join(RAW_DIR, local)
        if os.path.exists(dest) and not force:
            continue
        if verbose:
            print(f"  fetching {remote} ...", flush=True)
        blob = _download(remote, sha256)
        data = _csv_from_tar(blob, local) if remote.endswith(".tar.gz") else blob
        with open(dest + ".part", "wb") as f:
            f.write(data)
        os.replace(dest + ".part", dest)
    return RAW_DIR


# --------------------------------------------------------------------------- #
# Parsed trace
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class GenTD26:
    """
    All monitoring series on one shared grid of N_STEPS x 57 s.

    workload      per-step gateway totals (qps_total, qps_generative, qps_api,
                  active_containers, busy_gpus, queue_size, queue_rt_ms)
    pod_ids       hashed container ids, column order of the matrices below
    gpu_util      [steps, pods] GPU duty cycle, 0-100 %      (NaN = pod not running)
    mem_util      [steps, pods] container memory utilisation, 0-1
    queue_rt_ms   [steps, pods] gateway queueing time        (NaN = no request)
    latency_ms    [steps, pods] end-to-end inference latency (NaN = no request)
    """
    workload: pd.DataFrame
    pod_ids: List[str]
    gpu_util: np.ndarray
    mem_util: np.ndarray
    queue_rt_ms: np.ndarray
    latency_ms: np.ndarray
    step_seconds: int = STEP_SECONDS

    @property
    def n_steps(self) -> int:
        return len(self.workload)

    def series(self, name: str = "qps_total") -> np.ndarray:
        return self.workload[name].to_numpy(dtype=float)

    def replica_capacity(self) -> float:
        """
        Load one replica can carry, read off the trace instead of assumed:
        the least-squares slope of gateway QPS against the number of fully
        busy GPUs, i.e. QPS served per saturated GPU pod.
        """
        q = self.series("qps_total")
        g = self.series("busy_gpus")
        busy = g > 0.5
        return float(np.dot(q[busy], g[busy]) / np.dot(g[busy], g[busy]))

    def tenant_pool(self, min_latency_samples: int = 100) -> List[int]:
        """Pods observed for the whole day with enough latency samples to replay."""
        whole_day = ~np.isnan(self.gpu_util).any(axis=0)
        enough = (~np.isnan(self.latency_ms)).sum(axis=0) >= min_latency_samples
        return [int(i) for i in np.flatnonzero(whole_day & enough)]

    def slowdown(self) -> np.ndarray:
        """
        [steps, pods] observed throughput loss on each pod, in [0, 1).

        A request that takes (1 + x) times the pod's unloaded latency is being
        served at 1 / (1 + x) of full speed, so the loss is x / (1 + x). The
        unloaded latency is the pod's own lower-quartile latency. Steps where
        the pod served nothing count as no slowdown.
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)   # pods that never served a request
            base = np.nanpercentile(self.latency_ms, 25, axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            inflation = np.clip(self.latency_ms / base - 1.0, 0.0, None)
        loss = inflation / (1.0 + inflation)
        return np.nan_to_num(loss, nan=0.0)


def _read(name: str) -> pd.DataFrame:
    path = os.path.join(RAW_DIR, name)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} is missing. Download the trace first:  python -m traces.gentd26")
    return pd.read_csv(path)


def _pivot(df: pd.DataFrame, t0: float, pods: List[str]) -> np.ndarray:
    df = df.dropna(subset=["container_ip"])
    step = np.rint((df["timestamp_anon"].to_numpy() - t0) / STEP_SECONDS).astype(int)
    wide = (df.assign(step=step).groupby(["step", "container_ip"])["value"].mean()
              .unstack("container_ip"))
    return wide.reindex(index=range(N_STEPS), columns=pods).to_numpy(dtype=float)


@lru_cache(maxsize=1)
def load() -> GenTD26:
    """Parse the cached CSVs onto the common 57 s grid (takes about 2 s)."""
    gpu = _read("pod_gpu_duty_cycle_anon.csv")
    t0 = float(gpu["timestamp_anon"].min())
    pods = sorted(gpu["container_ip"].dropna().unique())

    gpu_util = _pivot(gpu, t0, pods)
    mem_util = _pivot(_read("pod_memory_util_anon.csv"), t0, pods)
    queue_rt = _pivot(_read("queue_rt_raw_anon.csv"), t0, pods)
    latency = _pivot(_read("pipeline_inference_data_anon.csv"), t0, pods)

    qps = _read("qps.csv")
    qps["step"] = np.rint((qps["timestamp_anon"] - t0) / STEP_SECONDS).astype(int)
    by_type = qps.groupby(["step", "request_type"])["value"].sum().unstack("request_type")
    generative = qps[qps["request_type"] == "Generative Requests"]
    qsize = _read("queue_size_raw_anon.csv")
    qsize["step"] = np.rint((qsize["timestamp_anon"] - t0) / STEP_SECONDS).astype(int)

    idx = pd.RangeIndex(N_STEPS, name="step")
    workload = pd.DataFrame(index=idx)
    # the gateway reports nothing when no request arrived: that is zero load
    workload["qps_generative"] = by_type.get("Generative Requests").reindex(idx).fillna(0.0)
    workload["qps_api"] = by_type.get("API Requests").reindex(idx).fillna(0.0)
    workload["qps_total"] = workload["qps_generative"] + workload["qps_api"]
    workload["active_containers"] = (generative.groupby("step")["container_ip"].nunique()
                                     .reindex(idx).fillna(0).astype(int))
    workload["busy_gpus"] = np.nansum(gpu_util, axis=1) / 100.0
    workload["queue_size"] = qsize.groupby("step")["value"].mean().reindex(idx).fillna(0.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)       # windows with no queued request
        workload["queue_rt_ms"] = pd.Series(np.nanmean(queue_rt, axis=1), index=idx)
    return GenTD26(workload, pods, gpu_util, mem_util, queue_rt, latency)


def request_arrivals(freq: str = "1h") -> pd.DataFrame:
    """
    The 23-day request log aggregated to a regular series.

    Columns: requests (arrivals per bin) and offered_load (mean number of
    requests in service at once = sum of execution seconds / bin seconds).
    """
    log = _read("lora_request_trace.csv")
    when = pd.to_datetime(log["gmt_create"])
    bins = pd.date_range(when.min().floor(freq), when.max().floor(freq), freq=freq)
    grouped = log.assign(bin=when.dt.floor(freq)).groupby("bin")
    out = pd.DataFrame(index=bins)
    out["requests"] = grouped.size().reindex(bins).fillna(0).astype(int)
    seconds = pd.Timedelta(freq).total_seconds()
    out["offered_load"] = grouped["exec_time_seconds"].sum().reindex(bins).fillna(0.0) / seconds
    return out


if __name__ == "__main__":
    fetch()
    trace = load()
    w = trace.workload
    print(f"GenTD26: {trace.n_steps} steps x {trace.step_seconds}s, {len(trace.pod_ids)} pods, "
          f"{len(trace.tenant_pool())} usable as node tenants")
    print(f"  gateway QPS   mean={w.qps_total.mean():.2f}  peak={w.qps_total.max():.2f}")
    print(f"  busy GPUs     mean={w.busy_gpus.mean():.2f}  peak={w.busy_gpus.max():.2f}")
    print(f"  per-replica capacity read from the trace = {trace.replica_capacity():.3f} QPS")
    arr = request_arrivals("1h")
    print(f"  request log   {len(arr)} hourly bins, {int(arr.requests.sum()):,} requests")
