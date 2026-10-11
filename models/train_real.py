"""
models/train_real.py  (Person A)  --  train the load LSTM on real data.

The Review-2 LSTM was trained on synthetic bursty series. This script trains
the same network on the GenTD26 request log instead: 23 days of real request
arrivals, binned at four resolutions and cut into overlapping 200-step
windows. Each window is z-scored, so the network learns the shape of real
load (daily ramps, lulls, bursts), not its scale.

The request log is a different recording from the one-day monitoring trace
the system is evaluated on, so the evaluation day is never seen in training.

    python -m models.train_real          # writes models/artifacts/lstm_gentd26.pt
"""
from __future__ import annotations

import time
from typing import List

import numpy as np

from models.forecasters import LSTM_REAL_PATH, train_lstm
from traces import gentd26

BIN_WIDTHS = ("5min", "10min", "15min", "30min")
WINDOW = 200
STRIDE = 100


def request_log_windows() -> List[np.ndarray]:
    """Overlapping windows of request arrivals at each bin width."""
    windows = []
    for width in BIN_WIDTHS:
        arrivals = gentd26.request_arrivals(width)["requests"].to_numpy(dtype=float)
        for start in range(0, len(arrivals) - WINDOW, STRIDE):
            w = arrivals[start:start + WINDOW]
            if w.std() > 0:
                windows.append(w)
    return windows


def main(epochs: int = 8, seed: int = 11) -> str:
    gentd26.fetch()
    windows = request_log_windows()
    print(f"training on {len(windows)} windows of the GenTD26 request log "
          f"({', '.join(BIN_WIDTHS)} bins, {WINDOW} steps each)")
    t0 = time.time()
    _, backend = train_lstm(series=windows, epochs=epochs, seed=seed,
                            save_path=LSTM_REAL_PATH, verbose=1)
    print(f"saved {LSTM_REAL_PATH}  [{backend}]  in {time.time() - t0:.0f}s")
    return LSTM_REAL_PATH


if __name__ == "__main__":
    main()
