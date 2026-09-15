"""
Fusion package for FAP-Scale framework.
"""

from .fusion_engine import FusionEngine, compute_node_score, calculate_replicas

__all__ = ["FusionEngine", "compute_node_score", "calculate_replicas"]
