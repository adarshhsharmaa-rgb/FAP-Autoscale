"""
Co-location Interference Matrix (Person B implementation).

Maintains a 15-bin interference matrix representing cross-workload resource contention
(Last-Level Cache, Memory Bandwidth, I/O bandwidth) with Exponential Moving Average (EMA) updates.

On real traces the 15 bins are workload profiles: 3 load patterns (periodic,
bursty, hybrid) x 5 utilisation levels (see `profile_bin`). Entry [i, j] is the
throughput a workload of profile i loses when it shares a host with profile j,
learned online from the latency inflation observed in the trace.
"""

from bisect import bisect_right
from typing import Union, Optional
import hashlib
import numpy as np


class InterferenceMatrix:
    """
    Co-location Interference Matrix covering 15 workload/bin profiles.

    Supports indexing by workload type strings (e.g., 'periodic', 'bursty', 'hybrid', 'cpu_heavy', 'io_heavy')
    or integer bin IDs (0 to 14).
    """

    WORKLOAD_TYPES = ["periodic", "bursty", "hybrid", "cpu_heavy", "io_heavy"]

    # Workload profiles for real traces: pattern x utilisation level = 15 bins.
    PATTERNS = ("periodic", "bursty", "hybrid")
    LEVEL_EDGES = (1.0, 10.0, 30.0, 60.0)   # % utilisation: idle | light | moderate | busy | saturated
    LEVEL_NAMES = ("idle", "light", "moderate", "busy", "saturated")

    def __init__(self, alpha: float = 0.1, seed: int = 42, prior: Optional[float] = None):
        """
        Initialize 15-bin interference matrix with realistic co-location degradation weights.

        Args:
            alpha: EMA smoothing coefficient (default 0.1).
            seed: Random seed for initial matrix seed.
            prior: If given, every entry starts at this value instead of the
                seeded random matrix, so nothing is assumed before the first
                observation and the matrix is learned purely from the trace.
        """
        self.alpha = alpha
        self.num_bins = 15
        self.updates = np.zeros((self.num_bins, self.num_bins), dtype=int)
        if prior is not None:
            self.matrix = np.full((self.num_bins, self.num_bins), float(np.clip(prior, 0.0, 1.0)))
            return

        # Initialize 15x15 matrix seeded with domain degradation values
        # Diagonal elements (same workload co-located) have higher contention (~0.3 - 0.6)
        # Off-diagonals have varied resource sharing interference (~0.05 - 0.45)
        np.random.seed(seed)
        base = np.random.uniform(0.1, 0.4, size=(self.num_bins, self.num_bins))
        # Make symmetric
        self.matrix = 0.5 * (base + base.T)
        # Increase diagonal contention
        np.fill_diagonal(self.matrix, np.random.uniform(0.4, 0.7, size=self.num_bins))
        # Ensure values stay in [0, 1]
        self.matrix = np.clip(self.matrix, 0.0, 1.0)

    @classmethod
    def profile_bin(cls, pattern: str, utilization: float) -> int:
        """
        Bin (0-14) of a workload profile: load pattern x utilisation level.

        Args:
            pattern: 'periodic', 'bursty' or 'hybrid' (anything else counts as hybrid).
            utilization: The workload's recent utilisation in % (0-100).
        """
        p = cls.PATTERNS.index(pattern) if pattern in cls.PATTERNS else cls.PATTERNS.index("hybrid")
        level = bisect_right(cls.LEVEL_EDGES, float(utilization))
        return p * len(cls.LEVEL_NAMES) + level

    @classmethod
    def profile_name(cls, bin_id: int) -> str:
        n = len(cls.LEVEL_NAMES)
        return f"{cls.PATTERNS[bin_id // n]}/{cls.LEVEL_NAMES[bin_id % n]}"

    def _get_bin_index(self, type_identifier: Union[str, int]) -> int:
        """
        Convert string type or integer bin index to valid bin integer in [0, 14].
        """
        if isinstance(type_identifier, (int, np.integer)):
            return int(type_identifier) % self.num_bins
        elif isinstance(type_identifier, str):
            clean_type = type_identifier.lower().strip()
            if clean_type in self.WORKLOAD_TYPES:
                # Map standard workload types to bins 0..4
                idx = self.WORKLOAD_TYPES.index(clean_type)
                return idx
            elif clean_type.startswith("bin_") or clean_type.startswith("bin"):
                try:
                    num = int(clean_type.replace("bin_", "").replace("bin", ""))
                    return num % self.num_bins
                except ValueError:
                    pass
            # Deterministic hash fallback for arbitrary workload type strings
            return int(hashlib.md5(clean_type.encode()).hexdigest(), 16) % self.num_bins
        return 0

    def get_interference(self, type_i: Union[str, int], type_j: Union[str, int]) -> float:
        """
        Deliverable function: Query interference score IS between workload type_i and type_j.

        Args:
            type_i: Workload type or bin identifier for first workload.
            type_j: Workload type or bin identifier for second workload.

        Returns:
            IS: Interference score float in range [0.0, 1.0].
        """
        idx_i = self._get_bin_index(type_i)
        idx_j = self._get_bin_index(type_j)
        score = float(self.matrix[idx_i, idx_j])
        return min(1.0, max(0.0, score))

    def update_ema(
        self,
        type_i: Union[str, int],
        type_j: Union[str, int],
        observed_degradation: float
    ) -> float:
        """
        Update matrix element via Exponential Moving Average (EMA):
        IS_new = (1 - alpha) * IS_old + alpha * observed_degradation

        Args:
            type_i: Workload type or bin index for workload i.
            type_j: Workload type or bin index for workload j.
            observed_degradation: Measured slowdown/degradation ratio in [0, 1].

        Returns:
            Updated interference score IS for the pair.
        """
        idx_i = self._get_bin_index(type_i)
        idx_j = self._get_bin_index(type_j)

        old_val = self.matrix[idx_i, idx_j]
        obs = float(np.clip(observed_degradation, 0.0, 1.0))
        new_val = (1.0 - self.alpha) * old_val + self.alpha * obs

        # Maintain symmetry in co-location matrix
        self.matrix[idx_i, idx_j] = new_val
        self.matrix[idx_j, idx_i] = new_val
        self.updates[idx_i, idx_j] += 1
        if idx_i != idx_j:
            self.updates[idx_j, idx_i] += 1

        return float(new_val)


# Global instance for deliverable convenience function
_GLOBAL_INTERFERENCE_MATRIX = InterferenceMatrix()


def get_interference(
    type_i: Union[str, int],
    type_j: Union[str, int],
    matrix: Optional[InterferenceMatrix] = None
) -> float:
    """
    Deliverable function for Person B:
    get_interference(type_i, type_j) -> IS

    Args:
        type_i: Workload type string or bin integer for workload i.
        type_j: Workload type string or bin integer for workload j.
        matrix: Optional custom InterferenceMatrix instance.

    Returns:
        IS: Interference score float in range [0.0, 1.0].
    """
    mat = matrix if matrix is not None else _GLOBAL_INTERFERENCE_MATRIX
    return mat.get_interference(type_i, type_j)
