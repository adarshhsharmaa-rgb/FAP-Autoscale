"""
traces/http_range.py

A seekable, read-only file object backed by HTTP Range requests.

The Backblaze Drive Stats archives are 1-2 GB zips of ~90 daily CSVs. Handing
this object to ``zipfile.ZipFile`` lets us read the zip's central directory and
then pull only the daily members we need, instead of downloading and unpacking
the whole quarter (~12 GB on disk).
"""
from __future__ import annotations

import http.client
import time
import urllib.error
import urllib.request

USER_AGENT = "FAP-Scale-research/1.0 (university OS lab project)"

# Everything a flaky connection can throw mid-transfer. http.client.IncompleteRead
# (connection dropped part-way through a body) is an HTTPException, not an OSError.
NETWORK_ERRORS = (urllib.error.URLError, http.client.HTTPException, TimeoutError,
                  ConnectionError, OSError)


class HTTPRangeFile:
    """Minimal file-like object: seek / tell / read over HTTP Range requests."""

    def __init__(self, url: str, block_size: int = 8 << 20, retries: int = 6,
                 timeout: float = 90.0):
        self.url = url
        self.block_size = block_size
        self.retries = retries
        self.timeout = timeout
        self.bytes_fetched = 0
        self._pos = 0
        self._buf = b""
        self._buf_start = 0
        self.size = self._content_length()

    # -- plumbing ---------------------------------------------------------- #
    def _open(self, req):
        last = None
        for attempt in range(self.retries):
            try:
                return urllib.request.urlopen(req, timeout=self.timeout)
            except NETWORK_ERRORS as exc:
                last = exc
                time.sleep(min(2 ** attempt, 30))
        raise IOError(f"giving up on {self.url}: {last}")

    def _content_length(self) -> int:
        req = urllib.request.Request(self.url, method="HEAD",
                                     headers={"User-Agent": USER_AGENT})
        with self._open(req) as resp:
            if resp.headers.get("Accept-Ranges", "").lower() != "bytes":
                raise IOError(f"{self.url} does not support Range requests")
            return int(resp.headers["Content-Length"])

    def _fetch(self, start: int, length: int) -> bytes:
        end = min(start + length, self.size) - 1
        last = None
        for attempt in range(self.retries):
            req = urllib.request.Request(self.url, headers={
                "User-Agent": USER_AGENT, "Range": f"bytes={start}-{end}"})
            try:
                with self._open(req) as resp:
                    data = resp.read()
                if len(data) == end - start + 1:
                    self.bytes_fetched += len(data)
                    return data
                last = IOError(f"short read: {len(data)} of {end - start + 1} bytes")
            except NETWORK_ERRORS as exc:
                last = exc
            time.sleep(min(2 ** attempt, 30))
        raise IOError(f"range {start}-{end} of {self.url} failed: {last}")

    # -- file interface ---------------------------------------------------- #
    def seekable(self) -> bool:
        return True

    def readable(self) -> bool:
        return True

    def tell(self) -> int:
        return self._pos

    def seek(self, offset: int, whence: int = 0) -> int:
        if whence == 0:
            self._pos = offset
        elif whence == 1:
            self._pos += offset
        elif whence == 2:
            self._pos = self.size + offset
        else:
            raise ValueError(f"bad whence {whence}")
        self._pos = max(0, min(self._pos, self.size))
        return self._pos

    def read(self, n: int = -1) -> bytes:
        if n is None or n < 0:
            n = self.size - self._pos
        n = min(n, self.size - self._pos)
        if n <= 0:
            return b""
        lo = self._pos - self._buf_start
        if not (0 <= lo and lo + n <= len(self._buf)):
            # read ahead a full block so sequential member reads cost few requests
            self._buf = self._fetch(self._pos, max(n, self.block_size))
            self._buf_start = self._pos
            lo = 0
        out = self._buf[lo:lo + n]
        self._pos += len(out)
        return out

    def close(self) -> None:
        self._buf = b""

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
