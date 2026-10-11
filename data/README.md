# Datasets

FAP-Scale is evaluated on two official, publicly released datasets. No part of
the reported results is drawn from a random generator.

| | Alibaba GenTD26 | Backblaze Drive Stats |
|---|---|---|
| Release | `cluster-trace-v2026-GenAI`, Alibaba Open Cluster Trace Program | Q2 2026 quarterly release |
| Publisher | Alibaba Group | Backblaze, Inc. |
| Source | https://github.com/alibaba/clusterdata/tree/master/cluster-trace-v2026-GenAI | https://www.backblaze.com/cloud-storage/resources/hard-drive-test-data |
| Papers | Lin et al., SoCC '25; Lin et al. (FlexPipe), EuroSys '26 | quarterly Drive Stats reports |
| What it is | One production day of a Stable Diffusion serving system, sampled every 57 s, plus a 23-day request log | A daily S.M.A.R.T. snapshot of every drive in Backblaze's data centres, with a failure flag |
| Used for | Load, node utilisation, queueing, interference | Node health telemetry and node failures |
| Loader | `traces/gentd26.py` | `traces/backblaze.py` |

## Why these two

The brief was an official dataset from 2025 or 2026. Google has not published a
cluster trace since the 2019 Borg trace, so "Google Cloud" was not an option.
Alibaba's other 2026 traces (`cluster-trace-gpu-v2026`, `cluster-trace-v2026-maas`)
are hosted on Aliyun OSS, which was unreachable from our network, and are 350 GB
and 15 GB. GenTD26 is on GitHub, is small, and happens to be sampled at 57 s,
which is the 60-second control window FAP-Scale was designed around.

No public trace from any year records node failures together with workload, so
failures come from the one large, current, real failure dataset there is.

## Alibaba GenTD26

`python -m traces.gentd26` downloads seven files (about 5 MB) from a pinned
upstream commit into `data/raw/gentd26/` and verifies their SHA-256.

The Alibaba repository has no licence beyond "we encourage anyone to use the
traces for study or research purposes", so **nothing from it is committed
here**: `data/raw/` is git-ignored and every clone fetches its own copy. The
publisher anonymised the data (time shift, metric scaling, hashed container
ids) while preserving trends and correlations, so loads are in anonymised QPS
units, not real request counts.

| File | Rows | Used as |
|---|---|---|
| `qps.csv` | 24,627 | gateway QPS per container, summed per window = the load L(t) |
| `pod_gpu_duty_cycle_anon.csv` | 157,417 | GPU utilisation of the tenant on each node |
| `pod_memory_util_anon.csv` | 214,890 | memory utilisation of the tenant on each node |
| `queue_rt_raw_anon.csv` | 23,478 | gateway queue time, the queue term of the CPI |
| `queue_size_raw_anon.csv` | 1,434 | gateway backlog |
| `pipeline_inference_data_anon.csv` | 34,421 | end-to-end latency; its inflation over a pod's own baseline is the observed interference |
| `lora_request_trace.csv` | 26,823 | 23 days of requests; trains the LSTM (a different recording from the evaluation day) |

## Backblaze Drive Stats Q2 2026

The quarterly archive is a 2.09 GB zip of 91 daily CSVs with 197 columns (about
12 GB unpacked). `python -m traces.backblaze` reads it over HTTP range requests
and streams only what is needed, newest day first:

* the 61 days from 2026-05-01 to 2026-06-30 (21.5 million drive-days, a fleet
  of about 355,000 drives);
* every one of the 862 drives that failed in that window, all of their days;
* a deterministic 1.2 % hash sample of the drives that did not fail (4,239 drives).

The result, 276,416 drive-days, is committed as
`processed/backblaze_q2_2026_sample.csv.gz` (1.8 MB) so nobody has to repeat
the 1.5 GB transfer. Serial numbers are replaced by integer ids.
`processed/backblaze_q2_2026_sample.meta.json` records the build.

Backblaze's terms: the data is free to use, Backblaze must be cited as the
source, and the data itself may not be sold.

| Column | Source | Meaning |
|---|---|---|
| `drive_id`, `date`, `model` | | drive identity (anonymised) and day |
| `failure` | `failure` | 1 on the last day a drive was operational before it failed |
| `will_fail` | derived | the drive fails somewhere in the window |
| `temperature_c` | SMART 194 | drive temperature; its 10-reading slope is the TTS |
| `realloc_sectors` | SMART 5 | sectors remapped after errors |
| `pending_sectors` | SMART 197 | unstable sectors waiting to be remapped |
| `offline_uncorrectable` | SMART 198 | sectors that could not be read or written |
| `reported_uncorrectable` | SMART 187 | errors ECC could not recover |
| `command_timeouts` | SMART 188 | aborted operations |
| `crc_errors` | SMART 199 | interface (cable/link) errors |
| `power_on_hours` | SMART 9 | age in service |

Because healthy drives are sub-sampled, failures are about 80 times more common
in the sample than in the fleet. Ranking metrics are unaffected; fleet-level
precision is reported with inverse sampling weights.

## How the Review-1 signal list maps onto real data

| Planned signal | In the real data | Note |
|---|---|---|
| CPU utilisation | GPU duty cycle of the pod on the node | the serving system is GPU-bound |
| Run-queue length | gateway queue time | normalised by 1 s |
| Context-switch rate | not available | no public trace records it; the CPI is re-weighted over the other two terms |
| Temperature trend | SMART 194 | real |
| DIMM errors | not available | uncorrectable *media* errors (SMART 187/198) take the role of an ECC error counter |
| Disk health | SMART 5 / 197 / 198 / 187 | real |
| Network retransmissions | not available | interface CRC errors (SMART 199) are the closest recorded link-error counter |

## Joining the two

The two datasets describe different systems, so the simulated cluster is a
composition, described in `simulator/cluster.py`: each node hosts one GenTD26
pod (its utilisation and slowdown) and is backed by one Backblaze drive (its
health and, for some nodes, its failure). Drive telemetry is daily and the
workload trace covers one day, so 30 days of drive history are replayed over
the run. The number of failing nodes per run is a parameter: it is failure
injection using real failure trajectories, not an estimate of how often a node
fails in a day.
