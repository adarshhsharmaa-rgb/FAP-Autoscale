"""
Tests for the dataset loaders in traces/:
  - HTTPRangeFile, the seekable reader that pulls single days out of the
    2 GB Backblaze archive
  - the committed Backblaze Drive Stats Q2 2026 sample
  - the Alibaba GenTD26 loader

The GenTD26 tests are skipped until the trace has been downloaded
(python -m traces.gentd26); the rest run offline.
"""

import io
import unittest
import zipfile

import numpy as np
import pandas as pd

from traces import backblaze, gentd26
from traces.http_range import HTTPRangeFile

HAVE_GENTD26 = gentd26.is_available()
NEEDS_GENTD26 = "GenTD26 not downloaded yet: run  python -m traces.gentd26"

SMART_COUNTERS = ["realloc_sectors", "pending_sectors", "offline_uncorrectable",
                  "reported_uncorrectable", "command_timeouts", "crc_errors"]


class _RangeOverBytes(HTTPRangeFile):
    """HTTPRangeFile with the network replaced by an in-memory blob."""

    def __init__(self, blob: bytes, block_size: int = 64):
        self._blob = blob
        self.requests = 0
        super().__init__("memory://blob", block_size=block_size)

    def _content_length(self) -> int:
        return len(self._blob)

    def _fetch(self, start: int, length: int) -> bytes:
        self.requests += 1
        data = self._blob[start:min(start + length, len(self._blob))]
        self.bytes_fetched += len(data)
        return data


class TestHTTPRangeFile(unittest.TestCase):

    def test_seek_and_read_match_the_bytes(self):
        blob = bytes(range(256)) * 4
        f = _RangeOverBytes(blob)
        self.assertEqual(f.size, len(blob))
        f.seek(100)
        self.assertEqual(f.read(10), blob[100:110])
        f.seek(-5, 2)
        self.assertEqual(f.read(), blob[-5:])
        self.assertEqual(f.read(3), b"")

    def test_sequential_reads_reuse_the_block(self):
        f = _RangeOverBytes(bytes(1000), block_size=400)
        for _ in range(20):
            f.read(10)
        self.assertEqual(f.requests, 1)

    def test_zip_member_can_be_read_without_the_rest(self):
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("data/2026-06-29.csv", "date,failure\n2026-06-29,0\n" * 500)
            zf.writestr("data/2026-06-30.csv", "date,failure\n2026-06-30,1\n" * 500)
        remote = _RangeOverBytes(buf.getvalue(), block_size=256)
        with zipfile.ZipFile(remote) as zf:
            text = zf.read("data/2026-06-30.csv").decode()
        self.assertTrue(text.startswith("date,failure\n2026-06-30,1"))
        self.assertLess(remote.bytes_fetched, len(buf.getvalue()) * 3)


class TestBackblazeSample(unittest.TestCase):

    def test_sample_is_real_2026_data(self):
        meta = backblaze.load_meta()
        self.assertEqual(meta["quarter"], "Q2_2026")
        self.assertTrue(meta["first_day"].startswith("2026-"))
        self.assertGreater(meta["failed_drives"], 500)
        self.assertGreater(meta["drive_days_scanned"], 10_000_000)

    def test_drive_days_schema_and_failure_bookkeeping(self):
        dd = backblaze.load_drive_days()
        for col in ["drive_id", "date", "failure", "will_fail", "temperature_c", "days_to_failure"] + SMART_COUNTERS:
            self.assertIn(col, dd.columns)
        self.assertEqual(dd.loc[dd["failure"] == 1, "drive_id"].nunique(), backblaze.load_meta()["failed_drives"])
        # the failure flag sits on a drive's last row and nowhere else
        failed = dd[dd["will_fail"]]
        last_rows = failed.groupby("drive_id").tail(1)
        self.assertTrue((last_rows["failure"] == 1).all())
        self.assertTrue((last_rows["days_to_failure"] == 0).all())
        self.assertTrue(dd.loc[~dd["will_fail"], "days_to_failure"].isna().all())

    def test_healthy_sample_is_deterministic(self):
        serials = pd.Series([f"SN{i:06d}" for i in range(20000)])
        a, b = backblaze._in_healthy_sample(serials), backblaze._in_healthy_sample(serials[::-1])
        self.assertTrue(np.array_equal(a, b[::-1]))
        self.assertAlmostEqual(a.mean(), backblaze.HEALTHY_SAMPLE_RATE, delta=0.004)


@unittest.skipUnless(HAVE_GENTD26, NEEDS_GENTD26)
class TestGenTD26(unittest.TestCase):

    def test_grid_and_workload(self):
        tr = gentd26.load()
        self.assertEqual(tr.n_steps, gentd26.N_STEPS)
        self.assertEqual(tr.gpu_util.shape, (gentd26.N_STEPS, len(tr.pod_ids)))
        w = tr.workload
        self.assertTrue((w["qps_total"] >= 0).all())
        self.assertTrue(np.allclose(w["qps_total"], w["qps_generative"] + w["qps_api"]))
        self.assertGreater(w["qps_total"].max(), 5 * w["qps_total"].median() / 5)

    def test_slowdown_is_a_fraction_and_zero_when_idle(self):
        tr = gentd26.load()
        s = tr.slowdown()
        self.assertTrue(((s >= 0) & (s < 1)).all())
        self.assertTrue((s[np.isnan(tr.latency_ms)] == 0).all())

    def test_capacity_and_tenants_come_from_the_trace(self):
        tr = gentd26.load()
        self.assertGreater(tr.replica_capacity(), 0)
        self.assertGreaterEqual(len(tr.tenant_pool()), 60)

    def test_request_log_covers_weeks(self):
        hourly = gentd26.request_arrivals("1h")
        self.assertGreater(len(hourly), 24 * 20)
        self.assertGreater(hourly["requests"].sum(), 20000)


if __name__ == "__main__":
    unittest.main(verbosity=2)
